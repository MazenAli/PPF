use ppf::distributions::*;

#[test]
fn test_normal_log_prob() {
    let normal = Normal::new(0.0, 1.0);
    let lp = normal.log_prob(0.0);
    assert!(lp < 0.0);
}
