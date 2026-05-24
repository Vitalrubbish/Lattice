use cudarc::driver::CudaSlice;
use std::collections::HashMap;

use crate::config::ModelConfig;

pub struct RawTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bytes: CudaSlice<u8>,
}

pub struct ModelWeights {
    pub cfg: ModelConfig,
    pub tensors: HashMap<String, RawTensor>,
}

impl ModelWeights {
    pub fn empty(cfg: &ModelConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            tensors: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, t: RawTensor) {
        self.tensors.insert(name, t);
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.bytes.len()).sum()
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.get(name)
    }
}
