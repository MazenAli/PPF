/// A probabilistic model that defines a log probability function over parameters.
///
/// This is the core abstraction for defining probabilistic models in PPF. A model encapsulates
/// the joint log probability of parameters and data, typically combining priors and likelihoods.
/// Models are used with MCMC samplers to perform Bayesian inference.
///
/// # Mathematical Background
///
/// A probabilistic model defines a log probability function over parameters θ:
/// ```text
/// log p(θ | data) ∝ log p(data | θ) + log p(θ)
/// ```
/// where:
/// - `log p(data | θ)` is the log likelihood
/// - `log p(θ)` is the log prior
/// - The proportionality constant is often omitted in MCMC
///
/// # Examples
///
/// ## Simple Prior Model
/// ```
/// use ppf::model::Model;
/// use ppf::distributions::{Distribution, Normal};
///
/// // Model with just a normal prior on one parameter
/// let prior_model = Model::new(|params: &[f64]| {
///     Normal::new(0.0, 1.0).log_prob(params[0])
/// });
/// ```
///
/// ## Bayesian Linear Regression
/// ```
/// use ppf::model::Model;
/// use ppf::distributions::{Distribution, Normal};
///
/// let x_data = vec![1.0, 2.0, 3.0, 4.0];
/// let y_data = vec![2.1, 3.9, 6.1, 8.0];
///
/// let model = Model::new(move |params: &[f64]| {
///     let intercept = params[0];
///     let slope = params[1];
///     let noise_std = params[2].exp(); // log-parameterized for positivity
///     
///     // Priors
///     let intercept_prior = Normal::new(0.0, 10.0).log_prob(intercept);
///     let slope_prior = Normal::new(0.0, 10.0).log_prob(slope);
///     let noise_prior = Normal::new(0.0, 1.0).log_prob(params[2]);
///     
///     // Likelihood
///     let likelihood: f64 = x_data.iter().zip(&y_data)
///         .map(|(&x, &y)| {
///             let predicted = intercept + slope * x;
///             Normal::new(predicted, noise_std).log_prob(y)
///         })
///         .sum();
///         
///     intercept_prior + slope_prior + noise_prior + likelihood
/// });
/// ```
pub struct Model<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// The log probability function that defines this model
    log_prob_fn: F,
}

impl<F> Model<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Create a new probabilistic model with the given log probability function.
    ///
    /// The function should accept a slice of parameter values and return the
    /// log probability (up to a constant) of those parameters given the model.
    /// This typically includes both prior and likelihood terms.
    ///
    /// # Arguments
    /// * `log_prob_fn` - A function `Fn(&[f64]) -> f64` that computes log probability
    ///
    /// # Returns
    /// A new `Model` instance wrapping the provided function
    ///
    /// # Notes
    /// - The function should return finite values for valid parameter regions
    /// - Returning `-f64::INFINITY` indicates invalid/impossible parameter values
    /// - The function will be called many times during MCMC sampling
    ///
    /// # Examples
    /// ```
    /// use ppf::model::Model;
    /// use ppf::distributions::{Distribution, Normal};
    ///
    /// // Bayesian inference for normal mean with known variance
    /// let data = vec![1.0, 2.0, 3.0];
    /// let model = Model::new(move |params: &[f64]| {
    ///     let mu = params[0];
    ///     
    ///     // Prior: N(0, 10²) - weakly informative
    ///     let prior = Normal::new(0.0, 10.0).log_prob(mu);
    ///     
    ///     // Likelihood: each observation ~ N(μ, 1²)
    ///     let likelihood: f64 = data.iter()
    ///         .map(|&x| Normal::new(mu, 1.0).log_prob(x))
    ///         .sum();
    ///         
    ///     prior + likelihood
    /// });
    /// ```
    pub fn new(log_prob_fn: F) -> Self {
        Model { log_prob_fn }
    }

    /// Evaluate the log probability of the model at the given parameter values.
    ///
    /// This method calls the underlying log probability function with the provided
    /// parameter values. The result represents the log probability (up to a constant)
    /// of the parameters under this model.
    ///
    /// # Arguments
    /// * `params` - A slice containing the parameter values to evaluate
    ///
    /// # Returns
    /// The log probability of the model evaluated at the given parameters
    ///
    /// # Examples
    /// ```
    /// use ppf::model::Model;
    /// use ppf::distributions::{Distribution, Normal};
    ///
    /// let model = Model::new(|params: &[f64]| {
    ///     Normal::new(0.0, 1.0).log_prob(params[0])
    /// });
    ///
    /// let log_prob = model.log_prob(&[0.0]);
    /// assert!(log_prob < 0.0); // Log probabilities are typically negative
    /// ```
    pub fn log_prob(&self, params: &[f64]) -> f64 {
        (self.log_prob_fn)(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    #[test]
    fn test_model_creation_and_evaluation() {
        let model = Model::new(|params: &[f64]| {
            let x = params[0];
            Normal::new(0.0, 1.0).log_prob(x)
        });

        let params = vec![0.0];
        let log_prob = model.log_prob(&params);

        // Should be close to the standard normal log pdf at 0
        let expected = Normal::new(0.0, 1.0).log_prob(0.0);
        assert!((log_prob - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_model() {
        let data = [1.0, 2.0, 3.0];
        let model = Model::new(move |params: &[f64]| {
            let mu = params[0];
            let prior = Normal::new(0.0, 10.0).log_prob(mu);
            let likelihood: f64 = data.iter().map(|&x| Normal::new(mu, 1.0).log_prob(x)).sum();
            prior + likelihood
        });

        // Test that we can evaluate the model
        let params = vec![2.0];
        let log_prob = model.log_prob(&params);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0); // Should be negative (log probability)
    }
}
