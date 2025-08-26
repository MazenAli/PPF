# Probabilistic Programming Framework (PPF)

A toy probabilistic programming library written in Rust for learning about Bayesian inference. This is a hobby project to explore basic concepts in probabilistic modeling and MCMC sampling.

## What's Included

- **Basic Distributions**: Normal, Exponential, and Inverse-Gamma with sampling and log-probability evaluation.
- **Simple Model Definition**: Define models using closures that return log-probabilities.
- **Metropolis-Hastings Sampler**: Basic MCMC implementation for posterior sampling.
- **Sample Analysis**: Compute means, standard deviations, and quantiles from MCMC output.
- **Examples**: A few working examples of Bayesian inference problems.

## Quick Start

```rust
use ppf::*;

// Define data
let data = [2.1, 1.9, 2.0, 2.2, 1.8];

// Create Bayesian model
let model = Model::new(move |params: &[f64]| {
    let mu = params[0];
    let prior = Normal::new(0.0, 10.0).log_prob(mu);
    let likelihood: f64 = data.iter()
        .map(|&x| Normal::new(mu, 1.0).log_prob(x))
        .sum();
    prior + likelihood
});

// Run MCMC inference
let mh = MetropolisHastings::new(1, 0.5);
let samples = mh.sample(&model, 10_000);

// Analyze results
println!("Posterior mean: {:.3}", samples.mean(0));
println!("95% CI: [{:.3}, {:.3}]", 
         samples.quantile(0, 0.025), samples.quantile(0, 0.975));
```

## Basic Structure

The library has four simple modules:

### Distributions (`ppf::distributions`)
Three basic probability distributions:
- **Normal**: Gaussian distribution 
- **Exponential**: Simple exponential distribution
- **InverseGamma**: Inverse-gamma distribution (useful for variance priors)

### Models (`ppf::model`)
Simple model wrapper that takes a closure:
```rust
let model = Model::new(|params: &[f64]| {
    // Return log probability given parameters
    prior_log_prob + likelihood_log_prob
});
```

### Samplers (`ppf::samplers`) 
Basic Metropolis-Hastings implementation for MCMC.

### Samples (`ppf::samples`)
Container for MCMC samples with basic statistics:
```rust
let posterior_mean = samples.mean(0);
let credible_interval = (samples.quantile(0, 0.025), samples.quantile(0, 0.975));
```

## Examples

### Basic Distribution Usage

```rust
use ppf::distributions::*;
use rand::rng;

// Sample from distributions
let normal = Normal::new(0.0, 1.0);
let exponential = Exponential::new(1.5);
let inv_gamma = InverseGamma::new(2.0, 1.0);

let mut rng = rng();
println!("Normal sample: {:.3}", normal.sample(&mut rng));
println!("Exponential sample: {:.3}", exponential.sample(&mut rng));
println!("InverseGamma sample: {:.3}", inv_gamma.sample(&mut rng));

// Evaluate log probabilities
println!("Normal log-prob at 0: {:.3}", normal.log_prob(0.0));
println!("Exponential log-prob at 1: {:.3}", exponential.log_prob(1.0));
```

### Available Examples

Run the examples to see the library in action:

```bash
# Simple Bayesian inference (unknown mean)
cargo run --example bayesian_inference

# Joint inference of mean and variance (exponential prior)
cargo run --example infer_mean_and_variance

# Joint inference with inverse-gamma prior on variance
cargo run --example infer_mean_variance_invgamma

# Basic distribution sampling
cargo run --example demo
```

## Building and Testing

```bash
# Build the project
cargo build

# Run tests
cargo test

# Check code style
cargo clippy
```

## Limitations

This is a learning project with many limitations:
- Only three basic distributions.
- Just one MCMC algorithm (Metropolis-Hastings).
- No convergence diagnostics beyond acceptance rate.
- Limited to continuous parameters.
- No automatic differentiation or gradient-based methods.
- Not optimized for performance.

For real work, consider mature libraries like PyMC, Stan, or Turing.jl.

## License

MIT License - see [LICENSE](LICENSE) file for details.