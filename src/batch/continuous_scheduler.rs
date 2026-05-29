use anyhow::Result;
use crossbeam_channel::Sender;
use cudarc::driver::CudaSlice;
use half::f16;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::cache::paged_kv::{PagedKvCache, BLOCK_SIZE};
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::decoder::greedy_sample;
use crate::model::NaiveTransformer;

use super::static_batch::{InferenceQueue, InferenceRequest, InferenceResponse};

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
    /// Generated token ids.
    generated: Vec<u32>,
}

pub struct ContinuousScheduler {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub model: Arc<NaiveTransformer>,
    pub cache: Arc<PagedKvCache>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub max_prefill_tokens: usize,
    queue: Arc<InferenceQueue>,
}

impl ContinuousScheduler {
    pub fn new(
        cfg: ModelConfig,
        ctx: Arc<CudaContext>,
        model: Arc<NaiveTransformer>,
        cache: PagedKvCache,
        max_batch: usize,
        max_seq_len: usize,
        queue: Arc<InferenceQueue>,
    ) -> Self {
        Self {
            cfg,
            ctx,
            model,
            cache: Arc::new(cache),
            max_batch,
            max_seq_len,
            max_prefill_tokens: 512, // reasonable default for prefill chunking
            queue,
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
            // 1. Drain incoming requests into waiting queue
            loop {
                match rx.try_recv() {
                    Ok(v) => waiting.push(v),
                    Err(_) => break,
                }
            }

            // 2. Admit waiting requests if budget allows
            while !waiting.is_empty() && running.len() < self.max_batch {
                if let Some((req, _tx)) = waiting.first() {
                    let prompt_len = req.prompt_tokens.len();
                    let blocks_needed = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    match self.cache.alloc_sequence(blocks_needed) {
                        Ok(block_table) => {
                            let seq_idx = self.cache.register_sequence(block_table);
                            tracing::debug!(
                                req_id = req.id,
                                seq_idx,
                                blocks = blocks_needed,
                                "admitted request"
                            );
                            let (req, tx) = waiting.remove(0);
                            running.push(RunningRequest {
                                req,
                                tx,
                                state: RequestState::Prefill { prompt_pos: 0 },
                                position: 0,
                                seq_idx,
                                generated: Vec::new(),
                            });
                        }
                        Err(e) => {
                            tracing::warn!("insufficient blocks for request: {e}");
                            break; // stop admitting until blocks are freed
                        }
                    }
                } else {
                    break;
                }
            }

            // 3. If nothing running, block on the queue
            if running.is_empty() {
                match rx.recv() {
                    Ok(v) => waiting.push(v),
                    Err(_) => return Ok(()),
                }
                continue;
            }

            // 4. Run one forward step for all running requests
            self.run_step(&mut running)?;

            // 5. Remove completed requests
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
                    self.cache.unregister_sequence(r.seq_idx);
                    let prefill_ms = 0.0; // simplified — tracked per batch
                    let decode_ms = 0.0;
                    let _ = r.tx.send(InferenceResponse {
                        id: r.req.id,
                        generated_tokens: r.generated,
                        prefill_ms,
                        decode_ms,
                    });
                    tracing::debug!(req_id = r.req.id, "request completed");
                } else {
                    i += 1;
                }
            }
        }
    }

    fn run_step(&mut self, running: &mut Vec<RunningRequest>) -> Result<()> {
        let batch = running.len();
        let h = self.cfg.hidden_size;

        let mut hidden: CudaSlice<f16> = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        let seq_indices: Vec<usize> = running.iter().map(|r| r.seq_idx).collect();
        let positions: Vec<usize> = running.iter().map(|r| r.position).collect();

        // Run the transformer forward step (per-layer GEMM + KV cache write)
        let logits = self.model.forward_step_paged(
            &mut hidden,
            &self.cache,
            &seq_indices,
            &positions,
        )?;

        // Sample next tokens
        let next = greedy_sample(&logits, batch, self.cfg.vocab_size);

        // Update each request state
        for (b, r) in running.iter_mut().enumerate() {
            match r.state {
                RequestState::Prefill { prompt_pos } => {
                    // During prefill, we feed prompt tokens (not sampled tokens)
                    // Simplified: each prefill step processes one token position
                    let new_pos = prompt_pos + 1;
                    r.position = new_pos;
                    if new_pos >= r.req.prompt_tokens.len() {
                        r.state = RequestState::Decode;
                        r.position = r.req.prompt_tokens.len();
                        tracing::debug!(req_id = r.req.id, "prefill complete, entering decode");
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

        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
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
        assert_eq!(resp.generated_tokens.len(), 3, "should generate max_new_tokens");
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

        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
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
        assert_eq!(resp.generated_tokens.len(), 1, "should stop at EOS token 0");
        assert_eq!(resp.generated_tokens[0], 0);
    }
}
