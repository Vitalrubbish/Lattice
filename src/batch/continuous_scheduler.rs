use anyhow::Result;
use crossbeam_channel::Sender;
use cudarc::driver::CudaSlice;
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::cache::fragmentation_tracker::RuntimeFragmentationTracker;
use crate::cache::paged_kv::BLOCK_SIZE;
use crate::cache::{EvictedSeqData, KvCacheBackend, PagedKvCache, SwapManager, advance_epoch, current_epoch};
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::decoder::greedy_sample;
#[cfg(feature = "kcmm")]
use crate::kcmm::KcmmPool;
use crate::model::Transformer;

use super::static_batch::{InferenceQueue, InferenceRequest, InferenceResponse};
use super::stats::StatsHandle;

// --- Cache backend enum ---

/// Owned handle that unifies the two cache backends.
pub enum CacheBackend {
    Baseline(Arc<PagedKvCache>),
    #[cfg(feature = "kcmm")]
    Kcmm(Arc<KcmmPool>),
}

impl CacheBackend {
    pub(crate) fn as_trait(&self) -> &dyn KvCacheBackend {
        match self {
            CacheBackend::Baseline(c) => c.as_ref(),
            #[cfg(feature = "kcmm")]
            CacheBackend::Kcmm(c) => c.as_ref(),
        }
    }

    pub(crate) fn is_kcmm(&self) -> bool {
        #[cfg(feature = "kcmm")]
        {
            matches!(self, CacheBackend::Kcmm(_))
        }
        #[cfg(not(feature = "kcmm"))]
        {
            false
        }
    }

    #[cfg(feature = "kcmm")]
    pub(crate) fn kcmm_pool(&self) -> Option<&KcmmPool> {
        match self {
            CacheBackend::Kcmm(p) => Some(p),
            CacheBackend::Baseline(_) => None,
        }
    }

    pub(crate) fn paged_kv(&self) -> Option<&PagedKvCache> {
        match self {
            CacheBackend::Baseline(c) => Some(c),
            #[cfg(feature = "kcmm")]
            CacheBackend::Kcmm(_) => None,
        }
    }
}

/// Maximum number of sequences allowed in the swapped queue.
const MAX_SWAPPED_SEQS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestState {
    /// Still consuming prompt tokens (with prefill chunking).
    Prefill { prompt_pos: usize },
    /// Generating new tokens one at a time.
    Decode,
}

struct RunningRequest {
    req: InferenceRequest,
    tx: Sender<InferenceResponse>,
    state: RequestState,
    /// Next token position to process (0-based, increments each step).
    position: usize,
    /// Index into PagedKvCache.seq_metadata.
    seq_idx: usize,
    /// Number of KV blocks allocated for this sequence.
    num_blocks: usize,
    /// Generated token ids.
    generated: Vec<u32>,
}

/// A sequence that has been evicted to host memory and awaits restoration.
struct SwappedRequest {
    request: InferenceRequest,
    tx: Sender<InferenceResponse>,
    generated: Vec<u32>,
    state: RequestState,
    position: usize,
    num_blocks: usize,
    kv_data: EvictedSeqData,
    /// Pool sequence index (only used in KCMM mode, set to 0 for baseline).
    seq_idx: usize,
}

pub struct ContinuousScheduler {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub model: Arc<dyn Transformer>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub max_prefill_tokens: usize,
    queue: Arc<InferenceQueue>,
    /// Unified cache backend enum (avoids downcasting).
    backend: CacheBackend,
    /// Swap-manager — only used in Baseline mode when swap is enabled.
    swap_manager: Option<SwapManager>,
    /// When true, Baseline mode will NOT use SwapManager — OOM simply
    /// skips the request.  This is the correct baseline for KCMM comparisons
    /// (the baseline should represent "no OS-level tiering support").
    disable_swap: bool,
    /// Swapped-out sequences waiting for GPU memory to become available.
    swapped: Vec<SwappedRequest>,
    /// Per-sequence last-access epoch for LRU victim selection.
    seq_last_epoch: HashMap<usize, u64>,
    /// Runtime fragmentation tracker — records snapshots at each scheduler step.
    tracker: RuntimeFragmentationTracker,
    /// Shared stats handle for the server to query fragmentation metrics.
    stats_handle: StatsHandle,
}

impl ContinuousScheduler {
    pub fn new(
        cfg: ModelConfig,
        ctx: Arc<CudaContext>,
        model: Arc<dyn Transformer>,
        max_batch: usize,
        max_seq_len: usize,
        queue: Arc<InferenceQueue>,
        stats_handle: StatsHandle,
        backend: CacheBackend,
        disable_swap: bool,
    ) -> Self {
        // bytes_per_token_elem = kv_heads × head_dim × 2 (one layer of K, f16)
        let bytes_per_token_elem = cfg.kv_heads() * cfg.head_dim() * 2;
        let tracker = RuntimeFragmentationTracker::new(bytes_per_token_elem);

        let swap_manager = if backend.is_kcmm() || disable_swap {
            None
        } else {
            Some(SwapManager::new())
        };

        Self {
            cfg,
            ctx: ctx.clone(),
            model,
            max_batch,
            max_seq_len,
            max_prefill_tokens: 512,
            queue,
            backend,
            swap_manager,
            disable_swap,
            swapped: Vec::new(),
            seq_last_epoch: HashMap::new(),
            tracker,
            stats_handle,
        }
    }

    pub fn spawn(mut self) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("continuous-scheduler".into())
            .spawn(move || {
                if let Err(e) = self.run() {
                    tracing::error!("continuous scheduler exit: {e:?}");
                }
            })
            .expect("spawn continuous scheduler")
    }

    fn run(&mut self) -> Result<()> {
        let rx = self.queue.receiver();
        let mut running: Vec<RunningRequest> = Vec::new();
        let mut waiting: Vec<(InferenceRequest, Sender<InferenceResponse>)> = Vec::new();

        loop {
            // -- Advance global epoch for LRU --
            let _epoch = advance_epoch();

            // 1. Drain incoming requests into waiting queue
            loop {
                match rx.try_recv() {
                    Ok(v) => waiting.push(v),
                    Err(_) => break,
                }
            }

            // 2. Admit waiting requests (with eviction if needed)
            self.admit_waiting(&mut running, &mut waiting);

            // 3. Try to restore swapped sequences after completions freed blocks
            self.try_restore_swapped(&mut running);

            // 4. If nothing running, block on the queue
            if running.is_empty() && self.swapped.is_empty() {
                match rx.recv() {
                    Ok(v) => waiting.push(v),
                    Err(_) => return Ok(()),
                }
                continue;
            }

            // If nothing running but there are swapped sequences, try restoring
            if running.is_empty() {
                self.try_restore_swapped(&mut running);
                if running.is_empty() {
                    match rx.recv() {
                        Ok(v) => waiting.push(v),
                        Err(_) => return Ok(()),
                    }
                    continue;
                }
            }

            // 5. Run one forward step for all running requests
            self.run_step(&mut running)?;

            // 5b. Record unified fragmentation snapshot after each step
            self.record_fragmentation_snapshot();

            // 6. Remove completed requests, attempt restoration
            self.drain_completed_swapped();
            let freed = self.remove_completed(&mut running);
            if freed > 0 {
                self.try_restore_swapped(&mut running);
            }
        }
    }

    /// Attempt to admit waiting requests, evicting running sequences if VRAM is full.
    fn admit_waiting(
        &mut self,
        running: &mut Vec<RunningRequest>,
        waiting: &mut Vec<(InferenceRequest, Sender<InferenceResponse>)>,
    ) {
        let mut i = 0;
        while i < waiting.len() && running.len() < self.max_batch {
            let prompt_len = waiting[i].0.prompt_tokens.len();
            let blocks_needed = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            match self.backend.as_trait().alloc_sequence(blocks_needed) {
                Ok(block_table) => {
                    let seq_idx = self.backend.as_trait().register_sequence(block_table);
                    self.seq_last_epoch
                        .insert(seq_idx, current_epoch());
                    tracing::debug!(
                        req_id = waiting[i].0.id,
                        seq_idx,
                        blocks = blocks_needed,
                        "admitted request"
                    );
                    let (req, tx) = waiting.remove(i);
                    running.push(RunningRequest {
                        req,
                        tx,
                        state: RequestState::Prefill { prompt_pos: 0 },
                        position: 0,
                        seq_idx,
                        num_blocks: blocks_needed,
                        generated: Vec::new(),
                    });
                    // No increment of i since we removed element at index i
                }
                Err(_e) => {
                    if free_blocks_available(self.backend.as_trait()) {
                        // Free blocks exist but alloc_sequence failed —
                        // likely a different error (not OOM). Skip this request.
                        tracing::warn!(
                            req_id = waiting[i].0.id,
                            "allocation failed despite free blocks, delaying request"
                        );
                        i += 1;
                        continue;
                    }

                    // VRAM exhausted — try to evict a running sequence

                    // KCMM path: use tiering engine instead of SwapManager
                    #[cfg(feature = "kcmm")]
                    if self.backend.is_kcmm() {
                        // Calculate how many blocks we need to free.
                        let free_blocks = self
                            .backend
                            .kcmm_pool()
                            .map(|p| p.free_physical_blocks())
                            .unwrap_or(0);
                        let needed_blocks =
                            blocks_needed.saturating_sub(free_blocks);

                        if needed_blocks == 0 {
                            // Shouldn't happen (alloc_sequence failed but free
                            // blocks exist), but handle gracefully.
                            i += 1;
                            continue;
                        }

                        // Collect block-granularity victims across potentially
                        // multiple sequences, respecting priority order.
                        let mut evicted_total = 0usize;
                        let mut victims_seen: usize = 0;
                        let max_victims = running.len().min(4); // at most 4 victims per attempt

                        while evicted_total < needed_blocks && victims_seen < max_victims {
                            let victim_idx = self.select_victim(running);
                            let idx = match victim_idx {
                                Some(i) => i,
                                None => break,
                            };
                            victims_seen += 1;

                            let victim = &running[idx];
                            let victim_seq_idx = victim.seq_idx;

                            // Collect GPU-resident block handles for this sequence
                            let candidates: Vec<_> = if let Some(pool) = self.backend.kcmm_pool() {
                                pool.sequence_gpu_handles(victim_seq_idx)
                                    .into_iter()
                                    .map(|(_, h)| h)
                                    .collect()
                            } else {
                                Vec::new()
                            };

                            if candidates.is_empty() {
                                // No GPU blocks to evict — skip this victim
                                continue;
                            }

                            let to_evict = (needed_blocks - evicted_total).min(candidates.len());

                            if let Some(pool) = self.backend.kcmm_pool() {
                                if let Some(ref tiering) = pool.tiering {
                                    match tiering.evict_blocks(
                                        pool,
                                        &candidates,
                                        to_evict,
                                    ) {
                                        Ok(evicted) => {
                                            evicted_total += evicted.len();
                                            tracing::debug!(
                                                req_id = victim.req.id,
                                                seq_idx = victim_seq_idx,
                                                evicted = evicted.len(),
                                                total_evicted = evicted_total,
                                                needed = needed_blocks,
                                                "KCMM: block-granularity eviction via tiering"
                                            );

                                            if evicted.len() >= candidates.len() {
                                                // Fully evicted: move to swapped
                                                let v = running.remove(idx);
                                                self.seq_last_epoch.remove(&v.seq_idx);
                                                let seq_idx = v.seq_idx;
                                                self.swapped.push(SwappedRequest {
                                                    request: v.req,
                                                    tx: v.tx,
                                                    generated: v.generated,
                                                    state: v.state,
                                                    position: v.position,
                                                    num_blocks: v.num_blocks,
                                                    kv_data: EvictedSeqData::dummy(),
                                                    seq_idx,
                                                });
                                            }
                                            // If partially evicted: sequence
                                            // stays in running with remaining
                                            // GPU-resident blocks.
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                req_id = victim.req.id,
                                                "KCMM eviction failed: {e}"
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        if evicted_total >= needed_blocks {
                            // Freed enough — retry admission
                            continue;
                        }

                        // Not enough blocks freed — skip this request
                        i += 1;
                        continue;
                    }

                    // Baseline path with SwapManager disabled:
                    // KCMM-comparison baseline — no eviction, OOM = skip request.
                    if self.disable_swap {
                        tracing::debug!(
                            req_id = waiting[i].0.id,
                            blocks_needed,
                            running = running.len(),
                            "baseline OOM (swap disabled): deferring request"
                        );
                        i += 1;
                        continue;
                    }

                    // Baseline path with SwapManager enabled (standalone server mode).
                    if let Some(victim_idx) = self.select_victim(running) {
                        let victim = &running[victim_idx];
                        tracing::info!(
                            req_id = victim.req.id,
                            seq_idx = victim.seq_idx,
                            blocks = victim.num_blocks,
                            "preempting sequence to free VRAM"
                        );

                        match self.swap_manager.as_ref().unwrap().evict_sequence(
                            self.backend.as_trait(),
                            victim.seq_idx,
                        ) {
                            Ok(kv_data) => {
                                let v = running.remove(victim_idx);
                                self.seq_last_epoch.remove(&v.seq_idx);
                                self.backend.as_trait().unregister_sequence(v.seq_idx);
                                tracing::debug!(
                                    req_id = v.req.id,
                                    "sequence evicted to host, will resume later"
                                );
                                self.swapped.push(SwappedRequest {
                                    request: v.req,
                                    tx: v.tx,
                                    generated: v.generated,
                                    state: v.state,
                                    position: v.position,
                                    num_blocks: v.num_blocks,
                                    kv_data,
                                    seq_idx: 0, // baseline: unused
                                });
                                // Now retry alloc_sequence for the waiting request
                                continue; // retry same i
                            }
                            Err(e) => {
                                tracing::error!(
                                    req_id = victim.req.id,
                                    "eviction failed: {e}"
                                );
                                i += 1;
                            }
                        }
                    } else {
                        // No sequence to evict — truly OOM
                        tracing::warn!(
                            req_id = waiting[i].0.id,
                            blocks_needed,
                            running = running.len(),
                            swapped = self.swapped.len(),
                            "cannot allocate blocks, no evictable sequences"
                        );
                        i += 1;
                    }
                }
            }

            // Safety valve: don't overflow the swapped queue
            if self.swapped.len() >= MAX_SWAPPED_SEQS {
                tracing::warn!(
                    swapped = self.swapped.len(),
                    "swapped queue full, deferring admissions"
                );
                break;
            }
        }
    }

    /// After completions freed blocks, try to restore evicted sequences.
    fn try_restore_swapped(&mut self, running: &mut Vec<RunningRequest>) {
        if self.swapped.is_empty() || running.len() >= self.max_batch {
            return;
        }

        let mut restored = Vec::new();

        for i in 0..self.swapped.len() {
            if running.len() >= self.max_batch {
                break;
            }

            let sw = &self.swapped[i];

            // KCMM path: use tiering engine to restore blocks in-place
            #[cfg(feature = "kcmm")]
            if self.backend.is_kcmm() {
                if let Some(pool) = self.backend.kcmm_pool() {
                    // Get the block table from the pool (sequence is still registered)
                    let block_table = pool.get_block_table(sw.seq_idx)
                        .unwrap_or_default();
                    if !block_table.is_empty() {
                        match pool.restore_evicted_blocks(&block_table) {
                            Ok(()) => {
                                // Mark as active again
                                pool.touch(sw.seq_idx);
                                self.seq_last_epoch
                                    .insert(sw.seq_idx, current_epoch());
                                tracing::debug!(
                                    req_id = sw.request.id,
                                    seq_idx = sw.seq_idx,
                                    position = sw.position,
                                    "KCMM: restored evicted sequence"
                                );

                                running.push(RunningRequest {
                                    req: sw.request.clone(),
                                    tx: sw.tx.clone(),
                                    state: sw.state,
                                    position: sw.position,
                                    seq_idx: sw.seq_idx,
                                    num_blocks: sw.num_blocks,
                                    generated: sw.generated.clone(),
                                });
                                restored.push(i);
                                continue;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    req_id = sw.request.id,
                                    seq_idx = sw.seq_idx,
                                    "KCMM: restore failed: {e}"
                                );
                                break;
                            }
                        }
                    }
                }
                break;
            }

            // Baseline path: SwapManager
            match self
                .swap_manager
                .as_ref()
                .unwrap()
                .restore_sequence(self.backend.as_trait(), &sw.kv_data)
            {
                Ok(new_block_table) => {
                    // Drop host-side buffers
                    self.swap_manager.as_ref().unwrap().drop_swapped(&sw.kv_data);

                    let seq_idx = self.backend.as_trait().register_sequence(new_block_table);
                    self.backend.as_trait().update_seq_len(seq_idx, sw.position);
                    self.seq_last_epoch
                        .insert(seq_idx, current_epoch());
                    tracing::debug!(
                        req_id = sw.request.id,
                        seq_idx,
                        position = sw.position,
                        "restored swapped sequence"
                    );

                    running.push(RunningRequest {
                        req: sw.request.clone(),
                        tx: sw.tx.clone(),
                        state: sw.state,
                        position: sw.position,
                        seq_idx,
                        num_blocks: sw.num_blocks,
                        generated: sw.generated.clone(),
                    });
                    restored.push(i);
                }
                Err(_e) => {
                    // Still no space — stop trying
                    break;
                }
            }
        }

        // Remove restored entries from swapped (in reverse order to preserve indices)
        for i in restored.into_iter().rev() {
            self.swapped.remove(i);
        }
    }

    /// Remove completed requests from running set.
    /// Returns the number of blocks freed (for triggering restoration).
    fn remove_completed(&mut self, running: &mut Vec<RunningRequest>) -> usize {
        let mut freed_blocks = 0usize;
        let mut i = 0;
        while i < running.len() {
            let r = &running[i];
            let is_done = match r.state {
                RequestState::Decode => {
                    let last_token = r.generated.last().copied();
                    last_token == Some(r.req.eos_token_id)
                        || r.generated.len() >= r.req.max_new_tokens
                        || r.position >= self.max_seq_len
                }
                RequestState::Prefill { .. } => false,
            };
            if is_done {
                let r = running.remove(i);
                freed_blocks += r.num_blocks;
                self.seq_last_epoch.remove(&r.seq_idx);
                // KCMM: cool the sequence so its blocks become preferred eviction targets.
                #[cfg(feature = "kcmm")]
                if let Some(pool) = self.backend.kcmm_pool() {
                    pool.cool(r.seq_idx);
                }
                self.backend.as_trait().unregister_sequence(r.seq_idx);
                let _ = r.tx.send(InferenceResponse {
                    id: r.req.id,
                    generated_tokens: r.generated,
                    prefill_ms: 0.0,
                    decode_ms: 0.0,
                });
                tracing::debug!(req_id = r.req.id, "request completed");
            } else {
                i += 1;
            }
        }
        freed_blocks
    }

    /// Check for completed swapped sequences (e.g., exceeded max_new_tokens or
    /// max_seq_len during swap). Send their responses with whatever was
    /// generated so far.
    fn drain_completed_swapped(&mut self) {
        let mut done = Vec::new();
        for i in 0..self.swapped.len() {
            let sw = &self.swapped[i];
            let is_done = match sw.state {
                RequestState::Decode => {
                    let last_token = sw.generated.last().copied();
                    last_token == Some(sw.request.eos_token_id)
                        || sw.generated.len() >= sw.request.max_new_tokens
                        || sw.position >= self.max_seq_len
                }
                RequestState::Prefill { .. } => false,
            };
            if is_done {
                done.push(i);
            }
        }

        for i in done.into_iter().rev() {
            let sw = self.swapped.remove(i);

            // KCMM: unregister the sequence from the pool (frees all blocks,
            // including CpuResident ones managed by the tiering engine).
            #[cfg(feature = "kcmm")]
            if self.backend.is_kcmm() {
                if let Some(pool) = self.backend.kcmm_pool() {
                    pool.unregister_sequence(sw.seq_idx);
                }
            }

            // Baseline: free host-side swap buffers.
            if !self.backend.is_kcmm() {
                if let Some(ref sm) = self.swap_manager {
                    sm.drop_swapped(&sw.kv_data);
                }
            }

            let _ = sw.tx.send(InferenceResponse {
                id: sw.request.id,
                generated_tokens: sw.generated,
                prefill_ms: 0.0,
                decode_ms: 0.0,
            });
            tracing::debug!(
                req_id = sw.request.id,
                "swapped sequence completed while evicted"
            );
        }
    }

    fn run_step(&mut self, running: &mut Vec<RunningRequest>) -> Result<()> {
        let batch = running.len();
        let h = self.cfg.hidden_size;

        let mut hidden: CudaSlice<f16> = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        let seq_indices: Vec<usize> = running.iter().map(|r| r.seq_idx).collect();
        let positions: Vec<usize> = running.iter().map(|r| r.position).collect();

        // Allocate additional blocks BEFORE forward step.
        for r in running.iter_mut() {
            let blocks_needed = (r.position / BLOCK_SIZE) + 1;
            while blocks_needed > r.num_blocks {
                match self.backend.as_trait().alloc_block() {
                    Ok(block_idx) => {
                        self.backend.as_trait()
                            .append_block_to_sequence(r.seq_idx, block_idx);
                        r.num_blocks += 1;
                    }
                    Err(_e) => {
                        tracing::warn!(
                            req_id = r.req.id,
                            position = r.position,
                            blocks = r.num_blocks,
                            "cannot grow KV cache, capping sequence"
                        );
                        break;
                    }
                }
            }
        }

        // Mark all running sequences as accessed at current epoch
        let epoch = current_epoch();
        for r in running.iter() {
            self.seq_last_epoch.insert(r.seq_idx, epoch);
            // KCMM: mark sequence as recently accessed for eviction policy
            #[cfg(feature = "kcmm")]
            if let Some(pool) = self.backend.kcmm_pool() {
                pool.touch(r.seq_idx);
            }
        }

        // Run the transformer forward step (per-layer GEMM + KV cache write)
        let token_ids: Vec<u32> = running.iter().map(|r| {
            match r.state {
                RequestState::Prefill { prompt_pos } => r.req.prompt_tokens[prompt_pos],
                RequestState::Decode => r.generated.last().copied().unwrap_or(0),
            }
        }).collect();
        let logits = self.model.forward_step_paged(
            &mut hidden,
            self.backend.as_trait(),
            &seq_indices,
            &token_ids,
            &positions,
        )?;

        // Sample next tokens
        let next = greedy_sample(&logits, batch, self.cfg.vocab_size);

        // Update each request state
        for (b, r) in running.iter_mut().enumerate() {
            match r.state {
                RequestState::Prefill { prompt_pos } => {
                    let new_pos = prompt_pos + 1;
                    r.position = new_pos;
                    // Blocks are pre-allocated for the full prompt length;
                    // report seq_len = full prompt_len for fragmentation
                    // tracking so IFR is not inflated during prefill ramp.
                    self.backend.as_trait()
                        .update_seq_len(r.seq_idx, r.req.prompt_tokens.len());
                    if new_pos >= r.req.prompt_tokens.len() {
                        r.state = RequestState::Decode;
                        r.position = r.req.prompt_tokens.len();
                        tracing::debug!(
                            req_id = r.req.id,
                            "prefill complete, entering decode"
                        );
                    } else {
                        r.state = RequestState::Prefill {
                            prompt_pos: new_pos,
                        };
                        r.position = new_pos;
                    }
                }
                RequestState::Decode => {
                    r.generated.push(next[b]);
                    r.position += 1;
                    self.backend.as_trait().update_seq_len(r.seq_idx, r.position);
                }
            }
        }

        Ok(())
    }

    /// Record a unified fragmentation snapshot from the current cache state
    /// and publish it to the shared stats handle for the server to query.
    fn record_fragmentation_snapshot(&mut self) {
        // Baseline mode: use the fragmentation tracker with the concrete PagedKvCache.
        // KCMM mode: KcmmPool tracks its own fragmentation internally.
        if let Some(paged) = self.backend.paged_kv() {
            self.tracker.record_unified(paged);
        }

        let snapshot = self.tracker.unified_summary();
        let latest_unified = self
            .tracker
            .unified_samples()
            .last()
            .copied();

        self.stats_handle.update_from_tracker(
            latest_unified,
            snapshot.sample_count,
            snapshot.rfi_avg,
            snapshot.rfi_peak,
            snapshot.rfi_stddev,
        );
    }

    /// Select a victim sequence from the running set for eviction.
    /// Returns the index in the running Vec, or None if no sequence can be evicted.
    ///
    /// Priority-aware LRU:
    ///   1. EVICTABLE sequences first (can be discarded without restore)
    ///   2. LOW priority sequences
    ///   3. NORMAL priority sequences
    ///   4. HIGH priority sequences (last resort)
    ///
    /// Within each priority tier, uses epoch-based LRU with block-count
    /// tiebreaker (prefer more blocks → fewer evictions to free memory).
    fn select_victim(&self, running: &[RunningRequest]) -> Option<usize> {
        if running.is_empty() {
            return None;
        }

        // Try each priority tier in order.
        // We fold into a single pass by scoring candidates.
        // Scoring: (priority_tier, epoch, -blocks) — lower = evict first.

        #[cfg(feature = "kcmm")]
        let pool = self.backend.kcmm_pool();

        let mut best: Option<(usize, u32, u64, isize)> = None; // (idx, prio_tier, epoch, -blocks)

        for (i, r) in running.iter().enumerate() {
            if matches!(r.state, RequestState::Prefill { .. }) {
                continue;
            }

            let epoch = self.seq_last_epoch.get(&r.seq_idx).copied().unwrap_or(0);
            let blocks = -(r.num_blocks as isize);

            // Determine priority tier (0 = evictable, 1 = low, 2 = normal, 3 = high)
            #[cfg(feature = "kcmm")]
            let prio_tier: u32 = {
                if let Some(ref p) = pool {
                    match p.sequence_priority(r.seq_idx) {
                        crate::kcmm::pool::SequencePriority::Evictable => 0,
                        crate::kcmm::pool::SequencePriority::Low => 1,
                        crate::kcmm::pool::SequencePriority::Normal => 2,
                        crate::kcmm::pool::SequencePriority::High => 3,
                    }
                } else {
                    2 // default normal
                }
            };
            #[cfg(not(feature = "kcmm"))]
            let prio_tier: u32 = 2; // normal

            match best {
                None => {
                    best = Some((i, prio_tier, epoch, blocks));
                }
                Some((_, best_tier, best_epoch, best_blocks)) => {
                    if prio_tier < best_tier
                        || (prio_tier == best_tier && epoch < best_epoch)
                        || (prio_tier == best_tier && epoch == best_epoch && blocks < best_blocks)
                    {
                        best = Some((i, prio_tier, epoch, blocks));
                    }
                }
            }
        }

        // Fall back to any running sequence if no decode-stage ones exist
        if best.is_none() {
            for (i, r) in running.iter().enumerate() {
                let epoch = self.seq_last_epoch.get(&r.seq_idx).copied().unwrap_or(0);
                let blocks = -(r.num_blocks as isize);
                #[cfg(feature = "kcmm")]
                let prio_tier: u32 = {
                    if let Some(ref p) = pool {
                        match p.sequence_priority(r.seq_idx) {
                            crate::kcmm::pool::SequencePriority::Evictable => 0,
                            crate::kcmm::pool::SequencePriority::Low => 1,
                            crate::kcmm::pool::SequencePriority::Normal => 2,
                            crate::kcmm::pool::SequencePriority::High => 3,
                        }
                    } else {
                        2
                    }
                };
                #[cfg(not(feature = "kcmm"))]
                let prio_tier: u32 = 2;

                match best {
                    None => {
                        best = Some((i, prio_tier, epoch, blocks));
                    }
                    Some((_, best_tier, best_epoch, best_blocks)) => {
                        if prio_tier < best_tier
                            || (prio_tier == best_tier && epoch < best_epoch)
                            || (prio_tier == best_tier && epoch == best_epoch && blocks < best_blocks)
                        {
                            best = Some((i, prio_tier, epoch, blocks));
                        }
                    }
                }
            }
        }

        best.map(|(i, _, _, _)| i)
    }
}

/// Check if there are free blocks available in the allocator pool.
fn free_blocks_available(cache: &dyn KvCacheBackend) -> bool {
    cache.has_free_blocks()
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cache::paged_kv::PagedKvCache;
    use crate::config::ModelConfig;
    use crate::cuda::CudaContext;
    use crate::model::{ModelWeights, NaiveTransformer};

    /// Small non-GQA config for fast integration tests.
    fn small_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            vocab_size: 1000,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
            torch_dtype: "float16".to_string(),
        }
    }

    #[test]
    fn e2e_continuous_single_request() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;
        let block_size = 16;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = PagedKvCache::new(
            ctx.clone(),
            cfg.clone(),
            max_batch,
            max_seq_len,
            block_size,
        )
        .expect("PagedKvCache");

        let queue = Arc::new(InferenceQueue::new());

        let stats = StatsHandle::new();
        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            max_batch,
            max_seq_len,
            queue.clone(),
            stats,
            CacheBackend::Baseline(Arc::new(cache)),
            true, // disable_swap: baseline for KCMM comparison
        );
        let _h = sched.spawn();

        // With zero weights, greedy_sample always picks token 0.
        let req = InferenceRequest {
            id: 100,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 3,
            eos_token_id: 2, // won't match token 0
        };

        let resp = queue.submit_blocking(req).expect("response");

        assert_eq!(resp.id, 100);
        assert_eq!(
            resp.generated_tokens.len(),
            3,
            "should generate max_new_tokens"
        );
        assert!(
            resp.generated_tokens.iter().all(|&t| t == 0),
            "zero weights produce token 0, got {:?}",
            resp.generated_tokens,
        );
        assert!(resp.prefill_ms >= 0.0);
        assert!(resp.decode_ms >= 0.0);
    }

    #[test]
    fn e2e_continuous_eos_termination() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;
        let block_size = 16;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = PagedKvCache::new(
            ctx.clone(),
            cfg.clone(),
            max_batch,
            max_seq_len,
            block_size,
        )
        .expect("PagedKvCache");

        let queue = Arc::new(InferenceQueue::new());

        let stats = StatsHandle::new();
        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            max_batch,
            max_seq_len,
            queue.clone(),
            stats,
            CacheBackend::Baseline(Arc::new(cache)),
            true, // disable_swap: baseline for KCMM comparison
        );
        let _h = sched.spawn();

        // Set eos_token_id=0 so decode terminates after 1 token.
        let req = InferenceRequest {
            id: 200,
            prompt_tokens: vec![5, 6],
            max_new_tokens: 100,
            eos_token_id: 0,
        };

        let resp = queue.submit_blocking(req).expect("response");
        assert_eq!(resp.id, 200);
        assert_eq!(
            resp.generated_tokens.len(),
            1,
            "should stop at EOS token 0"
        );
        assert_eq!(resp.generated_tokens[0], 0);
    }

    /// Test that the scheduler handles multiple requests correctly by
    /// submitting several requests and verifying all complete.
    #[test]
    fn e2e_multiple_requests() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;
        let block_size = 16;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = PagedKvCache::new(
            ctx.clone(),
            cfg.clone(),
            max_batch,
            max_seq_len,
            block_size,
        )
        .expect("PagedKvCache");

        let queue = Arc::new(InferenceQueue::new());

        let stats = StatsHandle::new();
        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            max_batch,
            max_seq_len,
            queue.clone(),
            stats,
            CacheBackend::Baseline(Arc::new(cache)),
            true, // disable_swap: baseline for KCMM comparison
        );
        let _h = sched.spawn();

        // Submit 4 requests concurrently
        let mut handles = Vec::new();
        for i in 0..4 {
            let q = queue.clone();
            handles.push(std::thread::spawn(move || {
                let req = InferenceRequest {
                    id: 300 + i,
                    prompt_tokens: vec![1, 2, 3],
                    max_new_tokens: 3,
                    eos_token_id: 2,
                };
                q.submit_blocking(req).expect("response")
            }));
        }

        for h in handles {
            let resp = h.join().unwrap();
            assert_eq!(resp.generated_tokens.len(), 3);
            assert!(resp.generated_tokens.iter().all(|&t| t == 0));
        }
    }

    /// Test the LRU victim selection logic in isolation.
    #[test]
    #[allow(unused_mut)]
    fn test_select_victim_logic() {
        // We can test select_victim by creating a scheduler with no cache needed
        // and directly calling select_victim on manually crafted RunningRequests.
        // Since select_victim only looks at seq_last_epoch and num_blocks,
        // we can test it without a real cache.
        let cfg = small_config();
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );
        let cache = PagedKvCache::new(ctx.clone(), cfg.clone(), 4, 64, 16)
            .expect("PagedKvCache");
        let queue = Arc::new(InferenceQueue::new());
        let stats = StatsHandle::new();

        let mut sched = ContinuousScheduler::new(
            cfg, ctx, model, 4, 64, queue, stats, CacheBackend::Baseline(Arc::new(cache)),
            true, // disable_swap: baseline for KCMM comparison
        );

        // Create mock running requests
        let req_base = InferenceRequest {
            id: 0,
            prompt_tokens: vec![1],
            max_new_tokens: 10,
            eos_token_id: 2,
        };
        let (tx, _rx) = crossbeam_channel::bounded(1);

        let running = vec![
            RunningRequest {
                req: InferenceRequest { id: 10, ..req_base.clone() },
                tx: tx.clone(),
                state: RequestState::Decode,
                position: 0,
                seq_idx: 100,
                num_blocks: 3,
                generated: vec![],
            },
            RunningRequest {
                req: InferenceRequest { id: 20, ..req_base.clone() },
                tx: tx.clone(),
                state: RequestState::Decode,
                position: 0,
                seq_idx: 200,
                num_blocks: 5,
                generated: vec![],
            },
            RunningRequest {
                req: InferenceRequest { id: 30, ..req_base.clone() },
                tx: tx.clone(),
                state: RequestState::Prefill { prompt_pos: 0 },
                position: 0,
                seq_idx: 300,
                num_blocks: 2,
                generated: vec![],
            },
        ];

        // All have epoch 0 (not in seq_last_epoch → default 0)
        // seq 100: 3 blocks, seq 200: 5 blocks
        // Both decode-stage, same epoch → larger block count (200) preferred
        let victim = sched.select_victim(&running);
        assert!(victim.is_some());
        let idx = victim.unwrap();
        assert_eq!(idx, 1, "should pick seq 200 (more blocks, same epoch)");

        // Give seq 100 a newer epoch → seq 200 (epoch 0) is now older
        sched.seq_last_epoch.insert(100, 10);
        let victim2 = sched.select_victim(&running);
        assert_eq!(victim2, Some(1), "should pick seq 200 (older epoch 0)");

        // Both decode sequences with newer epoch, victim should be seq 200 (epoch 0)
        sched.seq_last_epoch.insert(200, 5);
        let victim3 = sched.select_victim(&running);
        assert_eq!(victim3, Some(1), "should pick seq 200 (epoch 5 < epoch 10)");

        // No decode-stage → falls back to any sequence.
        // Reconstruct a prefill-only running list using index lookup.
        let victim4 = sched.select_victim(&running[2..3]);
        assert_eq!(victim4, Some(0), "should fall back to prefill-only sequence");

        // Empty running → None
        let victim5 = sched.select_victim(&[]);
        assert!(victim5.is_none());
    }

    /// Test that prefill-stage sequences are skipped during victim selection
    /// when decode-stage sequences exist.
    #[test]
    #[allow(unused_mut)]
    fn test_select_victim_skips_prefill() {
        let cfg = small_config();
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );
        let cache = PagedKvCache::new(ctx.clone(), cfg.clone(), 4, 64, 16)
            .expect("PagedKvCache");
        let queue = Arc::new(InferenceQueue::new());
        let stats = StatsHandle::new();

        let mut sched = ContinuousScheduler::new(
            cfg, ctx, model, 4, 64, queue, stats, CacheBackend::Baseline(Arc::new(cache)),
            true, // disable_swap: baseline for KCMM comparison
        );

        let req_base = InferenceRequest {
            id: 0,
            prompt_tokens: vec![1],
            max_new_tokens: 10,
            eos_token_id: 2,
        };
        let (tx, _rx) = crossbeam_channel::bounded(1);

        let running = vec![
            RunningRequest {
                req: InferenceRequest { id: 50, ..req_base.clone() },
                tx: tx.clone(),
                state: RequestState::Prefill { prompt_pos: 0 },
                position: 0,
                seq_idx: 50,
                num_blocks: 10, // many blocks, but in prefill → should be skipped
                generated: vec![],
            },
            RunningRequest {
                req: InferenceRequest { id: 60, ..req_base.clone() },
                tx: tx.clone(),
                state: RequestState::Decode,
                position: 0,
                seq_idx: 60,
                num_blocks: 1, // few blocks, but in decode → should be selected
                generated: vec![],
            },
        ];

        // seq 60 has epoch 0 (default), seq 50 has epoch 0 but is prefill
        let victim = sched.select_victim(&running);
        assert_eq!(victim, Some(1), "should pick decode-stage seq 60, not prefill seq 50");
    }
}
