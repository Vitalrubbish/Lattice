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

    pub fn tiny_llama() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: Some(4),
            vocab_size: 32000,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            torch_dtype: "bfloat16".to_string(),
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

// --- KCMM Configuration ---

/// Configuration for the KCMM (KV Cache Memory Manager) pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KcmmConfig {
    /// Tokens per block. Default: 16.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Maximum number of blocks in the pool. Default: 16384.
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,
    /// Path to the CPU swap buffer (typically in /dev/shm).
    /// Default: "/dev/shm/kcmm_swap".
    #[serde(default = "default_cpu_cache_path")]
    pub cpu_cache_path: String,
    /// Whether multi-tier storage (GPU→CPU→NVMe) is enabled.
    /// Default: true.
    #[serde(default = "default_tiering")]
    pub tiering: bool,
    /// Eviction policy: "lru", "lfu", or "fifo".
    /// Default: "lru".
    #[serde(default = "default_eviction_policy")]
    pub eviction_policy: String,
    /// Number of look-ahead blocks to prefetch per active sequence.
    /// Default: 4.
    #[serde(default = "default_prefetch_window")]
    pub prefetch_window: usize,
}

fn default_block_size() -> usize {
    16
}
fn default_max_blocks() -> usize {
    16384
}
fn default_cpu_cache_path() -> String {
    "/dev/shm/kcmm_swap".to_string()
}
fn default_tiering() -> bool {
    true
}
fn default_eviction_policy() -> String {
    "lru".to_string()
}
fn default_prefetch_window() -> usize {
    4
}

impl Default for KcmmConfig {
    fn default() -> Self {
        Self {
            block_size: default_block_size(),
            max_blocks: default_max_blocks(),
            cpu_cache_path: default_cpu_cache_path(),
            tiering: default_tiering(),
            eviction_policy: default_eviction_policy(),
            prefetch_window: default_prefetch_window(),
        }
    }
}
