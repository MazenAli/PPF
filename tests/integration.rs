use ppf::distributions::*;
use ppf::model::Model;
use ppf::samplers::*;

#[test]
fn test_complete_bayesian_workflow() {
    // This test verifies the complete workflow from model definition to posterior analysis
    let data = vec![2.1, 1.9, 2.0, 2.2, 1.8];
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;

    // Define Bayesian model: Normal likelihood with Normal prior
    let model = Model::new(move |params: &[f64]| {
        let mu = params[0];
        // Prior: weakly informative
        let prior = Normal::new(0.0, 5.0).log_prob(mu);
        // Likelihood: Normal with known variance
        let likelihood: f64 = data.iter().map(|&x| Normal::new(mu, 0.5).log_prob(x)).sum();
        prior + likelihood
    });

    // Run MCMC
    let mh = MetropolisHastings::new(1, 0.2);
    let samples = mh.sample(&model, 5000);

    // Verify we got the expected number of samples
    assert_eq!(samples.len(), 5000);
    assert_eq!(samples.n_params(), 1);

    // Check posterior is reasonable (should be close to data mean)
    let posterior_mean = samples.mean(0);
    assert!(
        (posterior_mean - data_mean).abs() < 0.5,
        "Posterior mean {:.3} should be close to data mean {:.3}",
        posterior_mean,
        data_mean
    );

    // Check we have reasonable uncertainty
    let posterior_std = samples.std(0);
    assert!(
        posterior_std > 0.1 && posterior_std < 1.0,
        "Posterior std {:.3} should be reasonable",
        posterior_std
    );
}

#[test]
fn test_model_with_multiple_parameters() {
    // Test a model with multiple parameters (mean and precision)
    let data = vec![1.0, 2.0, 1.5, 1.8, 2.2];
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;

    let model = Model::new(move |params: &[f64]| {
        let mu = params[0];
        let log_precision = params[1]; // log(1/sigma^2)
        let precision = log_precision.exp();
        let sigma = (1.0 / precision).sqrt();

        // Priors
        let mu_prior = Normal::new(0.0, 10.0).log_prob(mu);
        let precision_prior = Normal::new(0.0, 1.0).log_prob(log_precision); // log-normal prior on precision

        // Likelihood
        let likelihood: f64 = data
            .iter()
            .map(|&x| Normal::new(mu, sigma).log_prob(x))
            .sum();

        mu_prior + precision_prior + likelihood
    });

    // Test with 2 parameters
    let mh = MetropolisHastings::new(2, 0.1);
    let samples = mh.sample(&model, 2000);

    assert_eq!(samples.n_params(), 2);
    assert_eq!(samples.len(), 2000);

    // Basic sanity checks
    let mu_mean = samples.mean(0);
    assert!(
        (mu_mean - data_mean).abs() < 1.0,
        "Mean parameter estimate should be reasonable"
    );
}

#[test]
fn test_prior_only_model() {
    // Test a model with only priors (no data) - should sample from the prior
    let model = Model::new(|params: &[f64]| Normal::new(5.0, 1.0).log_prob(params[0]));

    let mh = MetropolisHastings::new(1, 0.5);
    let samples = mh.sample(&model, 3000);

    // Should be close to the prior mean
    let sample_mean = samples.mean(0);
    assert!(
        (sample_mean - 5.0).abs() < 0.3,
        "Prior-only model should sample close to prior mean, got {:.3}",
        sample_mean
    );

    // Should have approximately the prior standard deviation
    let sample_std = samples.std(0);
    assert!(
        (sample_std - 1.0).abs() < 0.3,
        "Prior-only model should have approximately prior std, got {:.3}",
        sample_std
    );
}

#[test]
fn test_samples_statistical_methods() {
    // Create deterministic samples to test statistical methods
    let mut samples = ppf::samples::Samples::new(2);

    // Add samples: [1,10], [2,20], [3,30], [4,40], [5,50]
    for i in 1..=5 {
        samples.push(vec![i as f64, (i * 10) as f64]);
    }

    // Test means
    assert_eq!(samples.mean(0), 3.0); // (1+2+3+4+5)/5
    assert_eq!(samples.mean(1), 30.0); // (10+20+30+40+50)/5

    // Test quantiles
    assert_eq!(samples.quantile(0, 0.0), 1.0); // min
    assert_eq!(samples.quantile(0, 1.0), 5.0); // max
    let median = samples.quantile(0, 0.5);
    assert!(median >= 3.0 && median <= 3.0); // median should be 3

    // Test get_param
    assert_eq!(samples.get_param(0), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(samples.get_param(1), vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn test_target_api_compatibility() {
    // This test exactly matches the target API from the original request
    let data = vec![5.0, 7.0, 4.0, 6.0];
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;

    // Define a model: Normal likelihood with unknown mean
    let model = Model::new(move |params: &[f64]| {
        let mu = params[0];
        let prior = Normal::new(0.0, 10.0).log_prob(mu);

        let likelihood: f64 = data.iter().map(|&x| Normal::new(mu, 1.0).log_prob(x)).sum();

        prior + likelihood
    });

    // Inference: run Metropolis-Hastings
    let mh = MetropolisHastings::new(1, 0.5); // 1 parameter, proposal std=0.5
    let samples = mh.sample(&model, 5000);

    // Verify the API works as expected
    let posterior_mean = samples.mean(0);

    // With this amount of data and weak prior, posterior should be close to data mean
    assert!(
        (posterior_mean - data_mean).abs() < 0.5,
        "Posterior mean should be close to data mean. Got {:.3}, expected ~{:.3}",
        posterior_mean,
        data_mean
    );

    // Verify we can compute other statistics
    let _posterior_std = samples.std(0);
    let _q025 = samples.quantile(0, 0.025);
    let _q975 = samples.quantile(0, 0.975);

    // Just verify these don't panic and return reasonable values
    assert!(samples.len() == 5000);
    assert!(samples.n_params() == 1);
}
