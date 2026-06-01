use anyhow::Result;
use crossbeam_channel::Sender;
use cudarc::driver::CudaSlice;
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::cache::fragmentation_tracker::RuntimeFragmentationTracker;
use crate::cache::paged_kv::{PagedKvCache, BLOCK_SIZE};
use crate::cache::{EvictedSeqData, SwapManager, advance_epoch, current_epoch};
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::decoder::greedy_sample;
use crate::model::Transformer;

use super::static_batch::{InferenceQueue, InferenceRequest, InferenceResponse};
use super::stats::StatsHandle;

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
}

pub struct ContinuousScheduler {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub model: Arc<dyn Transformer>,
    pub cache: Arc<PagedKvCache>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub max_prefill_tokens: usize,
    queue: Arc<InferenceQueue>,
    swap_manager: SwapManager,
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
        cache: PagedKvCache,
        max_batch: usize,
        max_seq_len: usize,
        queue: Arc<InferenceQueue>,
        stats_handle: StatsHandle,
    ) -> Self {
        // bytes_per_token_elem = kv_heads × head_dim × 2 (one layer of K, f16)
        let bytes_per_token_elem = cfg.kv_heads() * cfg.head_dim() * 2;
        let tracker = RuntimeFragmentationTracker::new(bytes_per_token_elem);

        Self {
            cfg,
            ctx: ctx.clone(),
            model,
            cache: Arc::new(cache),
            max_batch,
            max_seq_len,
            max_prefill_tokens: 512,
            queue,
            swap_manager: SwapManager::new(),
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

            match self.cache.alloc_sequence(blocks_needed) {
                Ok(block_table) => {
                    let seq_idx = self.cache.register_sequence(block_table);
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
                    if free_blocks_available(self.cache.as_ref()) {
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
                    if let Some(victim_idx) = self.select_victim(running) {
                        let victim = &running[victim_idx];
                        tracing::info!(
                            req_id = victim.req.id,
                            seq_idx = victim.seq_idx,
                            blocks = victim.num_blocks,
                            "preempting sequence to free VRAM"
                        );

                        match self.swap_manager.evict_sequence(
                            self.cache.as_ref(),
                            victim.seq_idx,
                        ) {
                            Ok(kv_data) => {
                                let v = running.remove(victim_idx);
                                self.seq_last_epoch.remove(&v.seq_idx);
                                self.cache.unregister_sequence(v.seq_idx);
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
            match self
                .swap_manager
                .restore_sequence(self.cache.as_ref(), &sw.kv_data)
            {
                Ok(new_block_table) => {
                    // Drop host-side buffers
                    self.swap_manager.drop_swapped(&sw.kv_data);

                    let seq_idx = self.cache.register_sequence(new_block_table);
                    self.cache.update_seq_len(seq_idx, sw.position);
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
                self.cache.unregister_sequence(r.seq_idx);
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
            self.swap_manager.drop_swapped(&sw.kv_data);
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
                match self.cache.alloc_block() {
                    Ok(block_idx) => {
                        self.cache
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
            &self.cache,
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
                    self.cache.update_seq_len(r.seq_idx, r.position);
                }
            }
        }

        Ok(())
    }

    /// Record a unified fragmentation snapshot from the current cache state
    /// and publish it to the shared stats handle for the server to query.
    fn record_fragmentation_snapshot(&mut self) {
        self.tracker.record_unified(&self.cache);

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
            self.tracker.average_ratio(),
            self.tracker.peak_ratio(),
            self.tracker.ratio_stddev(),
        );
    }

    /// Select a victim sequence from the running set for eviction.
    /// Returns the index in the running Vec, or None if no sequence can be evicted.
    ///
    /// Policy: LRU — the sequence with the smallest (oldest) epoch.
    /// Among sequences with the same epoch, prefer those with more blocks
    /// (fewer evictions needed to free a given amount of memory).
    fn select_victim(&self, running: &[RunningRequest]) -> Option<usize> {
        if running.is_empty() {
            return None;
        }

        // Don't preempt sequences still in Prefill.
        let mut best: Option<(usize, u64, isize)> = None; // (idx, epoch, -blocks)

        for (i, r) in running.iter().enumerate() {
            if matches!(r.state, RequestState::Prefill { .. }) {
                continue;
            }

            let epoch = self.seq_last_epoch.get(&r.seq_idx).copied().unwrap_or(0);
            let blocks = -(r.num_blocks as isize);

            match best {
                None => {
                    best = Some((i, epoch, blocks));
                }
                Some((_, best_epoch, best_blocks)) => {
                    if epoch < best_epoch
                        || (epoch == best_epoch && blocks < best_blocks)
                    {
                        best = Some((i, epoch, blocks));
                    }
                }
            }
        }

        // Fall back to any running sequence if no decode-stage ones exist
        if best.is_none() {
            for (i, r) in running.iter().enumerate() {
                let epoch = self.seq_last_epoch.get(&r.seq_idx).copied().unwrap_or(0);
                let blocks = -(r.num_blocks as isize);
                match best {
                    None => {
                        best = Some((i, epoch, blocks));
                    }
                    Some((_, best_epoch, best_blocks)) => {
                        if epoch < best_epoch
                            || (epoch == best_epoch && blocks < best_blocks)
                        {
                            best = Some((i, epoch, blocks));
                        }
                    }
                }
            }
        }

        best.map(|(i, _, _)| i)
    }
}

/// Check if there are free blocks available in the allocator pool.
fn free_blocks_available(cache: &PagedKvCache) -> bool {
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
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
            stats,
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
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
            stats,
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
}
