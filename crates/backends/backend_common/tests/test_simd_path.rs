#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_dispatch() {
    use herbert_backend_common::attention_common::dot_qf32_ki8;

    // On x86_64, the dispatch function should use AVX-512 or scalar
    let q = vec![1.0f32, 2.0, 3.0, 4.0];
    let k = vec![1i8, 2, 3, 4];
    let result = dot_qf32_ki8(&q, &k, 1.0);

    // Expected: 1*1 + 2*2 + 3*3 + 4*4 = 30.0
    assert!((result - 30.0).abs() < 0.001, "Expected 30.0, got {}", result);
}
