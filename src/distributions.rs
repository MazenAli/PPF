use rand::Rng;
use rand_distr::Distribution as RandDistribution;

/// Trait for probability distributions that can be sampled and evaluated.
///
/// This trait defines the core interface for probability distributions in the PPF library.
/// Implementations should provide both sampling capabilities and probability density/mass
/// function evaluation in log space.
pub trait Distribution {
    /// Sample a single value from the distribution.
    ///
    /// # Arguments
    /// * `rng` - A mutable reference to a random number generator
    ///
    /// # Returns
    /// A single random sample from the distribution
    ///
    /// # Example
    /// ```
    /// use ppf::distributions::{Distribution, Normal};
    /// use rand::rng;
    ///
    /// let normal = Normal::new(0.0, 1.0);
    /// let mut rng = rng();
    /// let sample = normal.sample(&mut rng);
    /// ```
    fn sample<R: Rng>(&self, rng: &mut R) -> f64;

    /// Compute the log probability density (or mass) at a given point.
    ///
    /// # Arguments
    /// * `x` - The value at which to evaluate the log probability
    ///
    /// # Returns
    /// The natural logarithm of the probability density function evaluated at `x`
    ///
    /// # Example
    /// ```
    /// use ppf::distributions::{Distribution, Normal};
    ///
    /// let normal = Normal::new(0.0, 1.0);
    /// let log_prob = normal.log_prob(0.0);
    /// assert!(log_prob < 0.0); // Log probabilities are typically negative
    /// ```
    fn log_prob(&self, x: f64) -> f64;
}

/// Normal (Gaussian) distribution with mean μ and standard deviation σ.
///
/// The normal distribution is a continuous probability distribution characterized by
/// its bell-shaped curve. It is parameterized by its mean (location) and standard
/// deviation (scale).
///
/// # Mathematical Definition
/// The probability density function is:
/// ```text
/// f(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
/// ```
///
/// # Examples
/// ```
/// use ppf::distributions::{Distribution, Normal};
/// use rand::rng;
///
/// // Standard normal distribution (mean=0, std=1)
/// let standard = Normal::new(0.0, 1.0);
///
/// // Sample from the distribution
/// let mut rng = rng();
/// let sample = standard.sample(&mut rng);
///
/// // Evaluate log probability
/// let log_prob = standard.log_prob(0.0);
/// ```
#[derive(Debug, Clone)]
pub struct Normal {
    /// Mean (location parameter) of the distribution
    mu: f64,
    /// Standard deviation (scale parameter) of the distribution  
    sigma: f64,
}

impl Normal {
    /// Create a new normal distribution with given mean and standard deviation.
    ///
    /// # Arguments
    /// * `mu` - The mean (location parameter) of the distribution
    /// * `sigma` - The standard deviation (scale parameter) of the distribution
    ///
    /// # Panics
    /// This function will panic if `sigma <= 0.0` as the standard deviation must be positive.
    ///
    /// # Examples
    /// ```
    /// use ppf::distributions::Normal;
    ///
    /// // Standard normal
    /// let standard = Normal::new(0.0, 1.0);
    ///
    /// // Normal with mean=5, std=2
    /// let custom = Normal::new(5.0, 2.0);
    /// ```
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(
            sigma > 0.0,
            "Standard deviation must be positive, got {}",
            sigma
        );
        Normal { mu, sigma }
    }

    /// Get the mean of the distribution.
    ///
    /// # Returns
    /// The mean (μ) parameter of the normal distribution
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation of the distribution.
    ///
    /// # Returns  
    /// The standard deviation (σ) parameter of the normal distribution
    pub fn std(&self) -> f64 {
        self.sigma
    }

    /// Get the variance of the distribution.
    ///
    /// # Returns
    /// The variance (σ²) of the normal distribution
    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
}

impl Distribution for Normal {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let z: f64 = rand_distr::Normal::new(0.0, 1.0).unwrap().sample(rng);
        self.mu + self.sigma * z
    }

    fn log_prob(&self, x: f64) -> f64 {
        let var = self.sigma * self.sigma;
        -0.5 * ((x - self.mu).powi(2) / var + var.ln() + (2.0 * std::f64::consts::PI).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_normal_sample_mean_variance() {
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rng();

        // Draw samples
        let n = 10_000;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(normal.sample(&mut rng));
        }

        // Compute sample mean and variance
        let mean: f64 = samples.iter().copied().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Check they're close to 0 and 1
        assert!(mean.abs() < 0.1, "mean too far: {}", mean);
        assert!((var - 1.0).abs() < 0.1, "variance too far: {}", var);
    }
}
