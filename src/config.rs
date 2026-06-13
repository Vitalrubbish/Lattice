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
    /// Maximum number of blocks to batch in a single eviction/restore
    /// operation.  Controls staging-buffer allocation size:
    /// `max_batch_blocks * block_bytes` bytes per staging buffer.
    /// Default: 64.
    #[serde(default = "default_max_batch_blocks")]
    pub max_batch_blocks: usize,
    /// Free block ratio below which the background eviction thread
    /// proactively evicts cold blocks to CPU.  Default: 0.2 (20%).
    #[serde(default = "default_low_watermark_threshold")]
    pub low_watermark_threshold: f32,
    /// Interval in milliseconds between background eviction checks.
    /// Default: 100.
    #[serde(default = "default_background_evict_interval_ms")]
    pub background_evict_interval_ms: u64,
    /// Number of attention sink blocks (initial tokens) protected from
    /// eviction by the "sink_window" policy.  Default: 1.
    #[serde(default = "default_attention_sink_blocks")]
    pub attention_sink_blocks: usize,
    /// Number of recent window blocks (final tokens) protected from
    /// eviction by the "sink_window" policy.  Default: 4.
    #[serde(default = "default_recent_window_blocks")]
    pub recent_window_blocks: usize,
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
fn default_max_batch_blocks() -> usize {
    64
}
fn default_low_watermark_threshold() -> f32 {
    0.2
}
fn default_background_evict_interval_ms() -> u64 {
    100
}
fn default_attention_sink_blocks() -> usize {
    1
}
fn default_recent_window_blocks() -> usize {
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
            max_batch_blocks: default_max_batch_blocks(),
            low_watermark_threshold: default_low_watermark_threshold(),
            background_evict_interval_ms: default_background_evict_interval_ms(),
            attention_sink_blocks: default_attention_sink_blocks(),
            recent_window_blocks: default_recent_window_blocks(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ModelConfig tests ---

    #[test]
    fn test_head_dim_llama_7b() {
        let cfg = ModelConfig::llama_7b_like();
        assert_eq!(cfg.head_dim(), 128); // 4096 / 32
    }

    #[test]
    fn test_head_dim_tiny_llama() {
        let cfg = ModelConfig::tiny_llama();
        assert_eq!(cfg.head_dim(), 64); // 2048 / 32
    }

    #[test]
    fn test_kv_heads_defaults_to_attention_heads() {
        let cfg = ModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: None,
            vocab_size: 1000,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
            torch_dtype: "float16".to_string(),
        };
        assert_eq!(cfg.kv_heads(), 8); // falls back to num_attention_heads
    }

    #[test]
    fn test_kv_heads_explicit_gqa() {
        let cfg = ModelConfig::tiny_llama();
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert_eq!(cfg.kv_heads(), 4); // explicit GQA value
    }

    #[test]
    fn test_llama_7b_constants() {
        let cfg = ModelConfig::llama_7b_like();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_position_embeddings, 4096);
    }

    #[test]
    fn test_tiny_llama_constants() {
        let cfg = ModelConfig::tiny_llama();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 5632);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_position_embeddings, 2048);
    }

    #[test]
    fn test_default_rope_theta() {
        assert_eq!(default_rope_theta(), 10000.0);
    }

    #[test]
    fn test_default_dtype() {
        assert_eq!(default_dtype(), "float16");
    }

    #[test]
    fn test_model_config_serde_roundtrip() {
        let cfg = ModelConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 8,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            vocab_size: 50000,
            max_position_embeddings: 2048,
            rope_theta: 500000.0,
            torch_dtype: "bfloat16".to_string(),
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let parsed: ModelConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.hidden_size, cfg.hidden_size);
        assert_eq!(parsed.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(parsed.num_attention_heads, cfg.num_attention_heads);
        assert_eq!(parsed.num_key_value_heads, cfg.num_key_value_heads);
        assert_eq!(parsed.vocab_size, cfg.vocab_size);
        assert_eq!(parsed.max_position_embeddings, cfg.max_position_embeddings);
        assert_eq!(parsed.rope_theta, cfg.rope_theta);
        assert_eq!(parsed.torch_dtype, cfg.torch_dtype);
    }

    #[test]
    fn test_model_config_serde_defaults() {
        // When num_key_value_heads and rope_theta are omitted, defaults should apply.
        let json = r#"{
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "max_position_embeddings": 256,
            "torch_dtype": "float16"
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.num_key_value_heads, None); // no GQA → None
        assert_eq!(cfg.rope_theta, 10000.0); // default
    }

    // --- KcmmConfig tests ---

    #[test]
    fn test_kcmm_config_defaults() {
        let cfg = KcmmConfig::default();
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.max_blocks, 16384);
        assert_eq!(cfg.cpu_cache_path, "/dev/shm/kcmm_swap");
        assert!(cfg.tiering);
        assert_eq!(cfg.eviction_policy, "lru");
        assert_eq!(cfg.prefetch_window, 4);
    }

    #[test]
    fn test_kcmm_config_serde_defaults() {
        let json = r#"{}"#;
        let cfg: KcmmConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.max_blocks, 16384);
        assert_eq!(cfg.cpu_cache_path, "/dev/shm/kcmm_swap");
        assert!(cfg.tiering);
        assert_eq!(cfg.eviction_policy, "lru");
        assert_eq!(cfg.prefetch_window, 4);
    }

    #[test]
    fn test_kcmm_config_serde_partial() {
        let json = r#"{
            "block_size": 32,
            "eviction_policy": "fifo",
            "tiering": false
        }"#;
        let cfg: KcmmConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.max_blocks, 16384); // default
        assert!(!cfg.tiering);
        assert_eq!(cfg.eviction_policy, "fifo");
        assert_eq!(cfg.prefetch_window, 4); // default
    }

    #[test]
    fn test_kcmm_config_serde_roundtrip() {
        let cfg = KcmmConfig {
            block_size: 64,
            max_blocks: 8192,
            cpu_cache_path: "/tmp/test_swap".to_string(),
            tiering: false,
            eviction_policy: "lfu".to_string(),
            prefetch_window: 8,
            max_batch_blocks: 32,
            low_watermark_threshold: 0.2,
            background_evict_interval_ms: 100,
            attention_sink_blocks: 1,
            recent_window_blocks: 4,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let parsed: KcmmConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.block_size, 64);
        assert_eq!(parsed.max_blocks, 8192);
        assert_eq!(parsed.cpu_cache_path, "/tmp/test_swap");
        assert!(!parsed.tiering);
        assert_eq!(parsed.eviction_policy, "lfu");
        assert_eq!(parsed.prefetch_window, 8);
        assert_eq!(parsed.max_batch_blocks, 32);
    }

    // --- ServerConfig defaults ---

    #[test]
    fn test_server_config_default_loader() {
        let json = r#"{
            "listen": "0.0.0.0:8080",
            "max_batch_size": 8,
            "max_seq_len": 2048,
            "model_path": "/tmp/model.safetensors"
        }"#;
        let cfg: ServerConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.loader, "read"); // default
        assert!(cfg.pipeline.is_none());
    }

    #[test]
    fn test_server_config_with_pipeline() {
        let json = r#"{
            "listen": "0.0.0.0:8080",
            "max_batch_size": 4,
            "max_seq_len": 1024,
            "model_path": "/tmp/model.safetensors",
            "pipeline": {
                "rank": 0,
                "world_size": 2,
                "next_addr": "10.0.0.2:8081",
                "listen_addr": "0.0.0.0:8080"
            }
        }"#;
        let cfg: ServerConfig = serde_json::from_str(json).expect("deserialize");
        let pipe = cfg.pipeline.expect("pipeline config");
        assert_eq!(pipe.rank, 0);
        assert_eq!(pipe.world_size, 2);
        assert_eq!(pipe.next_addr, Some("10.0.0.2:8081".to_string()));
        assert_eq!(pipe.listen_addr, "0.0.0.0:8080");
    }
}
