#[test]
fn test_dot_qf32_ki8_dispatch() {
    use herbert_backend_common::attention_common::dot_qf32_ki8;

    // Simple test: dot([1.0, 2.0, 3.0], [1i8, 2i8, 3i8]) with scale=1.0
    // Should be 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14.0
    let q = vec![1.0f32, 2.0, 3.0];
    let k = vec![1i8, 2, 3];
    let result = dot_qf32_ki8(&q, &k, 1.0);

    assert!((result - 14.0).abs() < 0.001, "Expected 14.0, got {}", result);
    println!("✓ dot_qf32_ki8 works: {} (expected 14.0)", result);
}

#[test]
fn test_sv_accum_i8_dispatch() {
    use herbert_backend_common::attention_common::sv_accum_i8;

    // Simple test: out += score * (v * v_scale)
    // out = [1.0, 2.0, 3.0], v = [1i8, 2i8, 3i8], score=2.0, v_scale=0.5
    // result = [1.0 + 2.0*0.5*1, 2.0 + 2.0*0.5*2, 3.0 + 2.0*0.5*3]
    //        = [1.0 + 1.0, 2.0 + 2.0, 3.0 + 3.0]
    //        = [2.0, 4.0, 6.0]
    let mut out = vec![1.0f32, 2.0, 3.0];
    let v = vec![1i8, 2, 3];
    sv_accum_i8(&mut out, &v, 2.0, 0.5);

    assert!((out[0] - 2.0).abs() < 0.001, "out[0]: expected 2.0, got {}", out[0]);
    assert!((out[1] - 4.0).abs() < 0.001, "out[1]: expected 4.0, got {}", out[1]);
    assert!((out[2] - 6.0).abs() < 0.001, "out[2]: expected 6.0, got {}", out[2]);
    println!("✓ sv_accum_i8 works: {:?} (expected [2.0, 4.0, 6.0])", out);
}
