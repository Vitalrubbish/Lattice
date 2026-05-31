use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
use std::collections::HashMap;

use crate::config::ModelConfig;

pub struct RawTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bytes: CudaSlice<u8>,
}

impl RawTensor {
    pub fn device_ptr(&self) -> u64 {
        *self.bytes.device_ptr()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn num_bytes(&self) -> usize {
        self.bytes.len()
    }
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

    pub fn layer_tensor(&self, layer: usize, suffix: &str) -> Option<&RawTensor> {
        let name = format!("model.layers.{layer}.{suffix}");
        self.get(&name)
    }

    pub fn try_get(&self, name: &str) -> anyhow::Result<&RawTensor> {
        self.get(name)
            .ok_or_else(|| anyhow::anyhow!("missing weight tensor: {name}"))
    }

    pub fn try_layer(&self, layer: usize, suffix: &str) -> anyhow::Result<&RawTensor> {
        self.layer_tensor(layer, suffix)
            .ok_or_else(|| {
                anyhow::anyhow!("missing weight tensor: model.layers.{layer}.{suffix}")
            })
    }
}
