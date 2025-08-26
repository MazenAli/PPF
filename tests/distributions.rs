use ppf::distributions::*;
use rand::rng;

#[test]
fn test_normal_distribution_properties() {
    let normal = Normal::new(5.0, 2.0);

    // Test basic properties
    assert!(normal.log_prob(5.0) > normal.log_prob(10.0)); // Higher prob at mean
    assert!(normal.log_prob(5.0) > normal.log_prob(0.0)); // Symmetric around mean

    // Test that log probabilities are negative (as they should be)
    assert!(normal.log_prob(5.0) < 0.0);
    assert!(normal.log_prob(0.0) < 0.0);
}

#[test]
fn test_normal_sampling_statistical_properties() {
    let normal = Normal::new(10.0, 3.0);
    let mut rng = rng();

    // Generate many samples
    let n_samples = 50_000;
    let samples: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();

    // Calculate sample statistics
    let sample_mean = samples.iter().sum::<f64>() / n_samples as f64;
    let sample_var = samples
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (n_samples - 1) as f64;

    // Verify sample mean is close to true mean
    assert!(
        (sample_mean - 10.0).abs() < 0.1,
        "Sample mean {:.3} should be close to true mean 10.0",
        sample_mean
    );

    // Verify sample variance is close to true variance (3^2 = 9)
    assert!(
        (sample_var - 9.0).abs() < 0.2,
        "Sample variance {:.3} should be close to true variance 9.0",
        sample_var
    );
}

#[test]
fn test_standard_normal_special_values() {
    let standard_normal = Normal::new(0.0, 1.0);

    // Test well-known values for standard normal
    let log_prob_at_zero = standard_normal.log_prob(0.0);
    let expected_log_prob = -0.5 * (2.0 * std::f64::consts::PI).ln(); // -log(sqrt(2π))

    assert!(
        (log_prob_at_zero - expected_log_prob).abs() < 1e-10,
        "Standard normal log prob at 0 should be -log(sqrt(2π))"
    );
}

#[test]
fn test_normal_distribution_symmetry() {
    let normal = Normal::new(0.0, 1.0);

    // Test symmetry around mean
    for x in [0.5, 1.0, 1.5, 2.0] {
        let log_prob_pos = normal.log_prob(x);
        let log_prob_neg = normal.log_prob(-x);
        assert!(
            (log_prob_pos - log_prob_neg).abs() < 1e-10,
            "Normal distribution should be symmetric around mean"
        );
    }
}

#[test]
fn test_normal_parameter_effects() {
    // Test mean parameter effect
    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(5.0, 1.0);

    assert!(normal1.log_prob(0.0) > normal1.log_prob(5.0));
    assert!(normal2.log_prob(5.0) > normal2.log_prob(0.0));

    // Test scale parameter effect
    let narrow = Normal::new(0.0, 0.5); // Smaller variance
    let wide = Normal::new(0.0, 2.0); // Larger variance

    // Narrow distribution should have higher prob at mean, lower prob at tails
    assert!(narrow.log_prob(0.0) > wide.log_prob(0.0)); // Higher at center
    assert!(narrow.log_prob(3.0) < wide.log_prob(3.0)); // Lower at tails
}
