//! MCMC sampling algorithms for Bayesian inference.
//!
//! This module provides implementations of Markov Chain Monte Carlo (MCMC) algorithms
//! for sampling from posterior distributions defined by probabilistic models.

use crate::distributions::{Distribution, Normal};
use crate::model::Model;
use crate::samples::Samples;
use rand::Rng;
use rand::rngs::ThreadRng;

/// Trait for MCMC sampling algorithms.
///
/// This trait defines the interface for MCMC samplers that can draw samples
/// from probabilistic models. Implementations should provide efficient sampling
/// strategies appropriate for different types of models and parameter spaces.
pub trait Sampler {
    /// Sample from the given model.
    ///
    /// # Arguments
    /// * `model` - The probabilistic model to sample from
    /// * `n_samples` - Number of samples to draw
    ///
    /// # Returns
    /// Samples from the model's posterior distribution
    fn sample<F>(&self, model: &Model<F>, n_samples: usize) -> Samples
    where
        F: Fn(&[f64]) -> f64;
}

/// Metropolis-Hastings sampler for MCMC inference.
///
/// The Metropolis-Hastings algorithm is a general-purpose MCMC method that can
/// sample from any continuous probability distribution. It works by proposing
/// new parameter values from a proposal distribution and accepting or rejecting
/// them based on the ratio of posterior probabilities.
///
/// # Algorithm
///
/// 1. Start with initial parameter values θ₀
/// 2. For each iteration i:
///    - Propose new values θ* from proposal distribution q(θ*|θᵢ₋₁)
///    - Compute acceptance probability α = min(1, p(θ*)/p(θᵢ₋₁))
///    - Accept θ* with probability α, otherwise keep θᵢ₋₁
///
/// # Tuning
///
/// The proposal standard deviation should be tuned to achieve acceptance rates
/// between 20-50% for good mixing. Too high acceptance leads to slow exploration,
/// too low acceptance leads to poor mixing.
///
/// # Examples
///
/// ```
/// use ppf::samplers::{Sampler, MetropolisHastings};
/// use ppf::model::Model;
/// use ppf::distributions::{Distribution, Normal};
///
/// let model = Model::new(|params: &[f64]| {
///     Normal::new(0.0, 1.0).log_prob(params[0])
/// });
///
/// let mh = MetropolisHastings::new(1, 0.5);
/// let samples = mh.sample(&model, 10000);
///
/// println!("Posterior mean: {:.3}", samples.mean(0));
/// ```
pub struct MetropolisHastings {
    /// Number of parameters in the model
    n_params: usize,
    /// Standard deviation for the proposal distribution
    proposal_std: f64,
    /// Random number generator
    rng: ThreadRng,
}

impl MetropolisHastings {
    /// Create a new Metropolis-Hastings sampler.
    ///
    /// # Arguments
    /// * `n_params` - Number of parameters in the model
    /// * `proposal_std` - Standard deviation for the proposal distribution
    ///
    /// # Example
    /// ```
    /// use ppf::samplers::MetropolisHastings;
    ///
    /// let mh = MetropolisHastings::new(1, 0.5);
    /// ```
    pub fn new(n_params: usize, proposal_std: f64) -> Self {
        MetropolisHastings {
            n_params,
            proposal_std,
            rng: rand::rng(),
        }
    }

    /// Propose a new state from the current state.
    fn propose(&mut self, current: &[f64]) -> Vec<f64> {
        let proposal_dist = Normal::new(0.0, self.proposal_std);
        current
            .iter()
            .map(|&x| x + proposal_dist.sample(&mut self.rng))
            .collect()
    }

    /// Compute the acceptance probability for a proposed move.
    fn acceptance_probability(&self, current_log_prob: f64, proposed_log_prob: f64) -> f64 {
        let log_alpha = proposed_log_prob - current_log_prob;
        log_alpha.exp().min(1.0)
    }
}

impl Sampler for MetropolisHastings {
    fn sample<F>(&self, model: &Model<F>, n_samples: usize) -> Samples
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut samples = Samples::new(self.n_params);
        let mut rng = rand::rng();

        // Initialize with zeros (could be made configurable)
        let mut current_state = vec![0.0; self.n_params];
        let mut current_log_prob = model.log_prob(&current_state);

        // If initial state has -inf log prob, try to find a better starting point
        if current_log_prob.is_infinite() && current_log_prob.is_sign_negative() {
            for _ in 0..100 {
                current_state = (0..self.n_params)
                    .map(|_| Normal::new(0.0, 1.0).sample(&mut rng))
                    .collect();
                current_log_prob = model.log_prob(&current_state);
                if current_log_prob.is_finite() {
                    break;
                }
            }
        }

        let mut n_accepted = 0;
        let mut mh_self = MetropolisHastings::new(self.n_params, self.proposal_std);

        for _ in 0..n_samples {
            // Propose new state
            let proposed_state = mh_self.propose(&current_state);
            let proposed_log_prob = model.log_prob(&proposed_state);

            // Accept or reject
            if proposed_log_prob.is_finite() {
                let alpha = mh_self.acceptance_probability(current_log_prob, proposed_log_prob);
                let u: f64 = rng.random();

                if u < alpha {
                    current_state = proposed_state;
                    current_log_prob = proposed_log_prob;
                    n_accepted += 1;
                }
            }

            // Store current state (whether it was accepted or not)
            samples.push(current_state.clone());
        }

        // Print acceptance rate for diagnostics
        let acceptance_rate = n_accepted as f64 / n_samples as f64;
        println!("Acceptance rate: {:.2}%", acceptance_rate * 100.0);

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metropolis_hastings_creation() {
        let mh = MetropolisHastings::new(2, 0.5);
        assert_eq!(mh.n_params, 2);
        assert_eq!(mh.proposal_std, 0.5);
    }

    #[test]
    fn test_proposal_generation() {
        let mut mh = MetropolisHastings::new(2, 0.1);
        let current = vec![1.0, 2.0];
        let proposed = mh.propose(&current);

        assert_eq!(proposed.len(), 2);
        // Proposals should be close to current values with small step size
        assert!((proposed[0] - 1.0).abs() < 1.0);
        assert!((proposed[1] - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_acceptance_probability() {
        let mh = MetropolisHastings::new(1, 0.5);

        // Better proposal should have high acceptance probability
        let alpha = mh.acceptance_probability(-2.0, -1.0);
        assert!(alpha > 0.5);

        // Worse proposal should have lower acceptance probability
        let alpha = mh.acceptance_probability(-1.0, -3.0);
        assert!(alpha < 0.5);

        // Equal proposals should have probability 1
        let alpha = mh.acceptance_probability(-1.0, -1.0);
        assert!((alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sampling_simple_model() {
        // Simple model: standard normal prior
        let model = Model::new(|params: &[f64]| Normal::new(0.0, 1.0).log_prob(params[0]));

        let mh = MetropolisHastings::new(1, 0.5);
        let samples = mh.sample(&model, 1000);

        assert_eq!(samples.len(), 1000);
        assert_eq!(samples.n_params(), 1);

        // Sample mean should be close to 0 with enough samples
        let mean = samples.mean(0);
        assert!(mean.abs() < 0.3, "Sample mean {} too far from 0", mean);
    }

    #[test]
    fn test_bayesian_inference() {
        // Bayesian inference: normal likelihood with known variance, unknown mean
        let data = [1.0, 1.5, 0.8, 1.2]; // Data centered around 1.0
        let data_mean = data.iter().sum::<f64>() / data.len() as f64;
        let model = Model::new(move |params: &[f64]| {
            let mu = params[0];
            // Prior: N(0, 10) - weakly informative
            let prior = Normal::new(0.0, 10.0).log_prob(mu);
            // Likelihood: each data point ~ N(mu, 1)
            let likelihood: f64 = data.iter().map(|&x| Normal::new(mu, 1.0).log_prob(x)).sum();
            prior + likelihood
        });

        let mh = MetropolisHastings::new(1, 0.3);
        let samples = mh.sample(&model, 2000);

        // Posterior mean should be close to sample mean
        let posterior_mean = samples.mean(0);

        // With weak prior, posterior should be close to data mean
        assert!(
            (posterior_mean - data_mean).abs() < 0.5,
            "Posterior mean {} not close to data mean {}",
            posterior_mean,
            data_mean
        );
    }
}
