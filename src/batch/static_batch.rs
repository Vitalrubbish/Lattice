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
