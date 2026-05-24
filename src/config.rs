use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_dtype")]
    pub torch_dtype: String,
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_dtype() -> String {
    "float16".to_string()
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn llama_7b_like() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            torch_dtype: "float16".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub listen: String,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub pipeline: Option<PipelineConfig>,
    pub model_path: PathBuf,
    #[serde(default = "default_loader")]
    pub loader: String,
}

fn default_loader() -> String {
    "read".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub rank: usize,
    pub world_size: usize,
    pub next_addr: Option<String>,
    pub listen_addr: String,
}
