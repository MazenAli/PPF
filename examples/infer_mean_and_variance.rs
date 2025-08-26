use ppf::distributions::*;
use ppf::model::Model;
use ppf::samplers::*;

fn main() {
    // Data: observed values from unknown normal distribution
    let data = [2.1, 1.8, 2.3, 1.9, 2.0, 2.4, 1.7, 2.2, 2.1, 1.6];

    // Calculate data statistics for comparison
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;
    let data_var =
        data.iter().map(|&x| (x - data_mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let data_std = data_var.sqrt();

    println!("Data mean: {:.3}, Data std: {:.3}", data_mean, data_std);

    // Define a model: Normal likelihood with unknown mean and variance
    // Parameters: [mu, log_sigma] where log_sigma enforces positivity
    let model = Model::new(move |params: &[f64]| {
        let mu = params[0];
        let sigma = params[1].exp(); // enforce positivity via exp transform

        let prior_mu = Normal::new(0.0, 10.0).log_prob(mu);
        let prior_sigma = Exponential::new(1.0).log_prob(sigma);

        let likelihood: f64 = data
            .iter()
            .map(|&x| Normal::new(mu, sigma).log_prob(x))
            .sum();

        prior_mu + prior_sigma + likelihood
    });

    // Inference: run Metropolis-Hastings
    println!("\nRunning MCMC inference...");
    let mh = MetropolisHastings::new(2, 0.1); // 2 parameters, modest proposal std
    let samples = mh.sample(&model, 12_000);

    // Analysis (skip burn-in)
    let burn_in = samples.len() / 4;
    println!(
        "Using {} samples after {} burn-in samples",
        samples.len() - burn_in,
        burn_in
    );

    // Transform log_sigma samples back to sigma scale
    let sigma_samples: Vec<f64> = samples.get_param(1).iter().map(|&x| x.exp()).collect();
    let posterior_sigma = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;

    // Posterior estimates
    let posterior_mu = samples.mean(0);
    let posterior_variance = posterior_sigma.powi(2);

    println!("\n=== POSTERIOR ESTIMATES ===");
    println!("Mean (μ): {:.3} ± {:.3}", posterior_mu, samples.std(0));
    println!("Std (σ): {:.3}", posterior_sigma);
    println!("Variance (σ²): {:.3}", posterior_variance);

    // Credible intervals
    println!("\n=== 95% CREDIBLE INTERVALS ===");
    println!(
        "Mean: [{:.3}, {:.3}]",
        samples.quantile(0, 0.025),
        samples.quantile(0, 0.975)
    );

    // Compute sigma credible interval
    let mut sorted_sigma = sigma_samples.clone();
    sorted_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma_q025 = sorted_sigma[(0.025 * sorted_sigma.len() as f64) as usize];
    let sigma_q975 = sorted_sigma[(0.975 * sorted_sigma.len() as f64) as usize];

    println!("Std: [{:.3}, {:.3}]", sigma_q025, sigma_q975);

    // Compare with data
    println!("\n=== COMPARISON WITH DATA ===");
    println!(
        "Data mean: {:.3} | Posterior mean: {:.3} | Difference: {:.3}",
        data_mean,
        posterior_mu,
        (data_mean - posterior_mu).abs()
    );
    println!(
        "Data std: {:.3} | Posterior std: {:.3} | Difference: {:.3}",
        data_std,
        posterior_sigma,
        (data_std - posterior_sigma).abs()
    );

    println!("\n=== PRIOR SPECIFICATION ===");
    println!("μ ~ Normal(0, 10)  [weakly informative]");
    println!("σ ~ Exponential(1) [rate=1, mean=1]");
    println!("Parameterization: [μ, log(σ)] for unconstrained sampling");
}
