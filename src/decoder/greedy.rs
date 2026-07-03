pub fn greedy_sample(logits: &[f32], batch: usize, vocab: usize) -> Vec<u32> {
    assert_eq!(logits.len(), batch * vocab);
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &logits[b * vocab..(b + 1) * vocab];
        let mut idx = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                idx = i as u32;
            }
        }
        out.push(idx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_per_row() {
        let l = vec![0.1, 0.9, 0.2, 5.0, 1.0, 2.0, -1.0, -2.0, 3.0];
        assert_eq!(greedy_sample(&l, 3, 3), vec![1u32, 0, 2]);
    }

    #[test]
    fn argmax_single_batch_single_vocab() {
        assert_eq!(greedy_sample(&[0.5], 1, 1), vec![0u32]);
    }

    #[test]
    fn argmax_all_negative() {
        let l = vec![-5.0, -1.0, -3.0];
        // -1.0 is the largest (closest to zero)
        assert_eq!(greedy_sample(&l, 1, 3), vec![1u32]);
    }

    #[test]
    fn argmax_tie_breaker_picks_first() {
        let l = vec![0.5, 0.5, 0.3];
        // First 0.5 at index 0 wins (ties pick lower index)
        assert_eq!(greedy_sample(&l, 1, 3), vec![0u32]);
    }

    #[test]
    fn argmax_tie_breaker_picks_first_vs_later() {
        // Index 1 and 3 both have value 1.0; index 1 should win
        let l = vec![0.1, 1.0, 0.3, 1.0, 0.2];
        assert_eq!(greedy_sample(&l, 1, 5), vec![1u32]);
    }

    #[test]
    fn argmax_multi_batch_tie_breaker() {
        let l = vec![
            0.1, 0.9, 0.9, // batch 0: tie at index 1,2 → pick 1
            0.5, 0.5, 0.5, // batch 1: all tie → pick 0
        ];
        assert_eq!(greedy_sample(&l, 2, 3), vec![1u32, 0u32]);
    }

    #[test]
    fn argmax_with_infinity() {
        let l = vec![1.0, f32::INFINITY, 2.0];
        assert_eq!(greedy_sample(&l, 1, 3), vec![1u32]);
    }

    #[test]
    fn argmax_with_negative_infinity() {
        let l = vec![f32::NEG_INFINITY, -5.0, f32::NEG_INFINITY];
        // -5.0 is the largest among all NEG_INF values
        assert_eq!(greedy_sample(&l, 1, 3), vec![1u32]);
    }

    #[test]
    fn argmax_with_nan_all_values_nan() {
        let l = vec![f32::NAN, f32::NAN, f32::NAN];
        // NaN comparisons always return false, so best stays NEG_INFINITY
        // and idx stays 0 (first element). This is deterministic but suboptimal.
        let result = greedy_sample(&l, 1, 3);
        assert_eq!(result, vec![0u32], "all-NaN should return index 0 deterministically");
    }

    #[test]
    fn argmax_with_nan_and_valid() {
        let l = vec![f32::NAN, 0.5, f32::NAN];
        // NaN comparison with any value is false, so 0.5 at index 1 wins.
        assert_eq!(greedy_sample(&l, 1, 3), vec![1u32]);
    }

    #[test]
    fn argmax_with_nan_first_then_valid() {
        let l = vec![f32::NAN, -1.0, -2.0];
        // NaN at index 0: v > NEG_INFINITY is false for NaN
        // -1.0 at index 1: -1.0 > NEG_INFINITY → true, best = -1.0, idx = 1
        // -2.0 at index 2: -2.0 > -1.0 → false
        // Result: index 1
        assert_eq!(greedy_sample(&l, 1, 3), vec![1u32]);
    }

    #[test]
    fn argmax_large_batch() {
        let batch = 128;
        let vocab = 256;
        let logits: Vec<f32> = (0..batch * vocab)
            .map(|i| (i % (vocab * batch)) as f32)
            .collect();
        let result = greedy_sample(&logits, batch, vocab);
        assert_eq!(result.len(), batch);
        // Each row: max is vocab-1 + batch_offset, but since we use linear fill,
        // the max in each batch row is the last element of that row (vocab-1).
        for b in 0..batch {
            let expected = (vocab - 1) as u32;
            assert_eq!(result[b], expected,
                "batch {}: expected max at index {}, got {}", b, expected, result[b]);
        }
    }
}
