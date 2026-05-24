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
}
