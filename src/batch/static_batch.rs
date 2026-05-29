use anyhow::Result;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use cudarc::driver::CudaSlice;
use half::f16;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::cache::KvCache;
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::decoder::greedy_sample;
use crate::model::NaiveTransformer;

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub id: u64,
    pub generated_tokens: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

pub struct InferenceQueue {
    sender: Sender<(InferenceRequest, Sender<InferenceResponse>)>,
    receiver: Receiver<(InferenceRequest, Sender<InferenceResponse>)>,
}

impl InferenceQueue {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }

    pub fn submit_blocking(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let (tx, rx) = bounded(1);
        self.sender.send((req, tx))?;
        Ok(rx.recv()?)
    }

    pub fn sender(&self) -> Sender<(InferenceRequest, Sender<InferenceResponse>)> {
        self.sender.clone()
    }
    pub fn receiver(&self) -> Receiver<(InferenceRequest, Sender<InferenceResponse>)> {
        self.receiver.clone()
    }
}

impl Default for InferenceQueue {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StaticScheduler {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub model: Arc<NaiveTransformer>,
    pub cache: KvCache,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    queue: Arc<InferenceQueue>,
}

impl StaticScheduler {
    pub fn new(
        cfg: ModelConfig,
        ctx: Arc<CudaContext>,
        model: Arc<NaiveTransformer>,
        cache: KvCache,
        max_batch_size: usize,
        max_seq_len: usize,
        queue: Arc<InferenceQueue>,
    ) -> Self {
        Self { cfg, ctx, model, cache, max_batch_size, max_seq_len, queue }
    }

    pub fn spawn(mut self) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("scheduler".into())
            .spawn(move || {
                if let Err(e) = self.run() {
                    tracing::error!("scheduler exit: {e:?}");
                }
            })
            .expect("spawn scheduler")
    }

    fn run(&mut self) -> Result<()> {
        let rx = self.queue.receiver();
        loop {
            let first = match rx.recv() {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };
            let mut reqs = vec![first];
            while reqs.len() < self.max_batch_size {
                match rx.try_recv() {
                    Ok(v) => reqs.push(v),
                    Err(_) => break,
                }
            }
            tracing::info!(n = reqs.len(), "batch start");
            self.run_one_batch(reqs)?;
        }
    }

    fn run_one_batch(
        &mut self,
        reqs: Vec<(InferenceRequest, Sender<InferenceResponse>)>,
    ) -> Result<()> {
        let batch = reqs.len();
        let h = self.cfg.hidden_size;

        let max_prompt = reqs.iter().map(|(r, _)| r.prompt_tokens.len()).max().unwrap_or(0);

        let allocator = self.cache.allocator();
        let mut slot_ids = Vec::with_capacity(batch);
        for _ in 0..batch {
            slot_ids.push(allocator.acquire().expect("slot"));
        }

        let mut hidden: CudaSlice<f16> = self.ctx.device.alloc_zeros::<f16>(batch * h)?;

        let t = Instant::now();
        self.model.prefill(&mut hidden, &mut self.cache, &slot_ids, max_prompt)?;
        let prefill_ms = t.elapsed().as_secs_f64() * 1e3;

        let max_new = reqs.iter().map(|(r, _)| r.max_new_tokens).max().unwrap_or(0);
        let mut outputs: Vec<Vec<u32>> = vec![Vec::new(); batch];
        let mut done = vec![false; batch];
        let mut positions: Vec<usize> = (0..batch).map(|b| reqs[b].0.prompt_tokens.len()).collect();

        let t = Instant::now();
        for _ in 0..max_new {
            if done.iter().all(|x| *x) {
                break;
            }
            let logits = self
                .model
                .forward_step(&mut hidden, &mut self.cache, &slot_ids, &positions)?;
            let next = greedy_sample(&logits, batch, self.cfg.vocab_size);
            for b in 0..batch {
                if done[b] {
                    continue;
                }
                outputs[b].push(next[b]);
                positions[b] += 1;
                if next[b] == reqs[b].0.eos_token_id
                    || outputs[b].len() >= reqs[b].0.max_new_tokens
                    || positions[b] >= self.max_seq_len
                {
                    done[b] = true;
                }
            }
        }
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;

        for &s in &slot_ids {
            allocator.release(s);
        }

        for (b, (req, tx)) in reqs.into_iter().enumerate() {
            let _ = tx.send(InferenceResponse {
                id: req.id,
                generated_tokens: std::mem::take(&mut outputs[b]),
                prefill_ms,
                decode_ms,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::config::ModelConfig;
    use crate::cuda::CudaContext;
    use crate::model::{ModelWeights, NaiveTransformer};

    /// Small non-GQA config for fast integration tests.
    /// kv_heads == num_attention_heads so KvCache.append_step assertion passes.
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
    fn e2e_single_request_lifecycle() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = KvCache::new(ctx.clone(), cfg.clone(), max_batch, max_seq_len).expect("cache");
        let queue = Arc::new(InferenceQueue::new());

        let sched = StaticScheduler::new(
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
        // Use eos_token_id != 0 so decode doesn't terminate immediately.
        let req = InferenceRequest {
            id: 42,
            prompt_tokens: vec![1, 2, 3, 4, 5],
            max_new_tokens: 5,
            eos_token_id: 2,
        };

        let resp = queue.submit_blocking(req).expect("response");

        assert_eq!(resp.id, 42);
        assert_eq!(resp.generated_tokens.len(), 5, "should generate max_new_tokens");
        assert!(
            resp.generated_tokens.iter().all(|&t| t == 0),
            "zero weights produce token 0, got {:?}",
            resp.generated_tokens
        );
        assert!(resp.prefill_ms >= 0.0, "prefill_ms should be non-negative");
        assert!(resp.decode_ms >= 0.0, "decode_ms should be non-negative");
    }

    #[test]
    fn e2e_batch_two_requests() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = KvCache::new(ctx.clone(), cfg.clone(), max_batch, max_seq_len).expect("cache");
        let queue = Arc::new(InferenceQueue::new());

        let sched = StaticScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
        );
        let _h = sched.spawn();

        // Submit two requests. The scheduler batches them together.
        let req1 = InferenceRequest {
            id: 1,
            prompt_tokens: vec![10, 20],
            max_new_tokens: 3,
            eos_token_id: 2,
        };
        let req2 = InferenceRequest {
            id: 2,
            prompt_tokens: vec![30, 40, 50],
            max_new_tokens: 4,
            eos_token_id: 2,
        };

        // Submit both before the scheduler wakes up, so they batch.
        let queue1 = queue.clone();
        let queue2 = queue.clone();
        let h1 = std::thread::spawn(move || queue1.submit_blocking(req1));
        let h2 = std::thread::spawn(move || queue2.submit_blocking(req2));

        let r1 = h1.join().unwrap().expect("resp1");
        let r2 = h2.join().unwrap().expect("resp2");

        assert_eq!(r1.id, 1);
        assert_eq!(r1.generated_tokens.len(), 3);
        assert!(r1.generated_tokens.iter().all(|&t| t == 0));

        assert_eq!(r2.id, 2);
        assert_eq!(r2.generated_tokens.len(), 4);
        assert!(r2.generated_tokens.iter().all(|&t| t == 0));
    }

    #[test]
    fn e2e_eos_termination() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = small_config();
        let max_batch = 4;
        let max_seq_len = 64;

        let weights = ModelWeights::empty(&cfg);
        let model = Arc::new(
            NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights).expect("model"),
        );

        let cache = KvCache::new(ctx.clone(), cfg.clone(), max_batch, max_seq_len).expect("cache");
        let queue = Arc::new(InferenceQueue::new());

        let sched = StaticScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model,
            cache,
            max_batch,
            max_seq_len,
            queue.clone(),
        );
        let _h = sched.spawn();

        // With zero weights, token 0 is always generated.
        // Set eos_token_id=0 to verify early termination works.
        let req = InferenceRequest {
            id: 99,
            prompt_tokens: vec![7, 8, 9],
            max_new_tokens: 100,
            eos_token_id: 0, // matches the zero-weight output, so stops after 1 token
        };

        let resp = queue.submit_blocking(req).expect("response");
        assert_eq!(resp.id, 99);
        assert_eq!(resp.generated_tokens.len(), 1, "should stop at EOS token 0");
        assert_eq!(resp.generated_tokens[0], 0);
    }
}
