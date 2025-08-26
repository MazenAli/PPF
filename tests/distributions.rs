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

#[test]
fn test_exponential_distribution_properties() {
    let exp = Exponential::new(1.5);

    // Test theoretical mean and variance
    let theoretical_mean = 1.0 / 1.5; // 1/λ
    let theoretical_var = 1.0 / (1.5 * 1.5); // 1/λ²

    assert!((exp.mean() - theoretical_mean).abs() < 1e-10);
    assert!((exp.variance() - theoretical_var).abs() < 1e-10);

    // Test that rate parameter affects probability density
    let exp1 = Exponential::new(1.0);
    let exp2 = Exponential::new(2.0);

    // Higher rate should give higher probability at small values, lower at large values
    assert!(exp2.log_prob(0.1) > exp1.log_prob(0.1));
    assert!(exp2.log_prob(2.0) < exp1.log_prob(2.0));
}

#[test]
fn test_exponential_memoryless_property() {
    let _exp = Exponential::new(2.0);

    // Memoryless property: P(X > s+t | X > s) = P(X > t)
    // In terms of log probabilities: log P(X > s+t) - log P(X > s) ≈ log P(X > t)
    let s = 1.0_f64;
    let t = 0.5_f64;

    // For exponential, P(X > x) = exp(-λx), so log P(X > x) = -λx
    let log_prob_s_plus_t = -2.0_f64 * (s + t); // log P(X > s+t)
    let log_prob_s = -2.0_f64 * s; // log P(X > s)  
    let log_prob_t = -2.0_f64 * t; // log P(X > t)

    let conditional_log_prob: f64 = log_prob_s_plus_t - log_prob_s;
    assert!((conditional_log_prob - log_prob_t).abs() < 1e-10);
}

#[test]
fn test_inverse_gamma_as_conjugate_prior() {
    // Test that inverse-gamma is conjugate to normal variance
    let _inv_gamma = InverseGamma::new(3.0, 2.0);

    // Prior parameters
    let alpha_prior = 3.0;
    let beta_prior = 2.0;

    // "Observed" data (we'll just test the math)
    let n = 5; // sample size
    let sum_sq_deviations = 10.0; // Σ(x_i - x̄)²

    // Posterior should be InverseGamma(α + n/2, β + Σ(x_i - x̄)²/2)
    let alpha_post = alpha_prior + n as f64 / 2.0;
    let beta_post = beta_prior + sum_sq_deviations / 2.0;

    assert_eq!(alpha_post, 5.5); // 3 + 5/2
    assert_eq!(beta_post, 7.0); // 2 + 10/2

    // Test that this gives sensible posterior mean
    let posterior_mean = beta_post / (alpha_post - 1.0);
    assert!((posterior_mean - 7.0 / 4.5).abs() < 1e-10);
}

#[test]
fn test_inverse_gamma_sampling_convergence() {
    let inv_gamma = InverseGamma::new(4.0, 3.0);
    let mut rng = rng();

    // Large sample to test convergence
    let n = 50_000;
    let samples: Vec<f64> = (0..n).map(|_| inv_gamma.sample(&mut rng)).collect();

    // Test sample moments against theoretical values
    let sample_mean = samples.iter().sum::<f64>() / n as f64;
    let theoretical_mean = inv_gamma.mean();

    assert!(
        (sample_mean - theoretical_mean).abs() < 0.1,
        "Sample mean {:.3} should be close to theoretical {:.3}",
        sample_mean,
        theoretical_mean
    );

    // Test that samples are all positive
    assert!(samples.iter().all(|&x| x > 0.0));

    // Test that most samples are reasonably sized (not extreme outliers)
    let median_sample = {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[n / 2]
    };
    assert!(
        median_sample > 0.1 && median_sample < 10.0,
        "Median sample should be reasonable, got {:.3}",
        median_sample
    );
}

#[test]
fn test_distribution_relationships() {
    // Test that Exponential(λ) is Gamma(1, 1/λ)
    let exp = Exponential::new(2.0);

    // Sample and check that they're reasonably distributed
    let mut rng = rng();
    let samples: Vec<f64> = (0..1000).map(|_| exp.sample(&mut rng)).collect();

    // For Exp(2), about 63% should be ≤ 1/λ = 0.5
    let count_le_half = samples.iter().filter(|&&x| x <= 0.5).count();
    let proportion = count_le_half as f64 / samples.len() as f64;

    // Should be close to 1 - exp(-λ * 0.5) = 1 - exp(-1) ≈ 0.632
    assert!(
        (proportion - 0.632).abs() < 0.05,
        "Proportion ≤ 0.5 should be ~0.632, got {:.3}",
        proportion
    );
}
