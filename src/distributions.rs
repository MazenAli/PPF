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

/// Exponential distribution with rate parameter λ.
///
/// The exponential distribution is a continuous probability distribution often used
/// to model waiting times or lifetimes. It is characterized by its memoryless property
/// and is commonly used as a prior for positive scale parameters.
///
/// # Mathematical Definition
/// The probability density function is:
/// ```text
/// f(x) = λ * exp(-λx) for x ≥ 0
/// ```
///
/// # Examples
/// ```
/// use ppf::distributions::{Distribution, Exponential};
/// use rand::rng;
///
/// // Exponential with rate 1.0 (mean = 1.0)
/// let exp = Exponential::new(1.0);
///
/// // Sample from the distribution
/// let mut rng = rng();
/// let sample = exp.sample(&mut rng);
///
/// // Evaluate log probability (only positive values allowed)
/// let log_prob = exp.log_prob(2.0);
/// ```
#[derive(Debug, Clone)]
pub struct Exponential {
    /// Rate parameter (λ) of the distribution
    rate: f64,
}

impl Exponential {
    /// Create a new exponential distribution with given rate parameter.
    ///
    /// # Arguments
    /// * `rate` - The rate parameter (λ) of the distribution, must be positive
    ///
    /// # Panics
    /// This function will panic if `rate <= 0.0` as the rate parameter must be positive.
    ///
    /// # Examples
    /// ```
    /// use ppf::distributions::Exponential;
    ///
    /// // Exponential with rate 1.0 (mean = 1.0)
    /// let exp1 = Exponential::new(1.0);
    ///
    /// // Exponential with rate 2.0 (mean = 0.5)
    /// let exp2 = Exponential::new(2.0);
    /// ```
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Rate parameter must be positive, got {}", rate);
        Exponential { rate }
    }

    /// Get the rate parameter of the distribution.
    ///
    /// # Returns
    /// The rate parameter (λ) of the exponential distribution
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Get the mean of the distribution.
    ///
    /// # Returns
    /// The mean (1/λ) of the exponential distribution
    pub fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    /// Get the variance of the distribution.
    ///
    /// # Returns
    /// The variance (1/λ²) of the exponential distribution
    pub fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
}

impl Distribution for Exponential {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Use inverse transform sampling: if U ~ Uniform(0,1), then -ln(U)/λ ~ Exp(λ)
        let u: f64 = rng.random();
        -u.ln() / self.rate
    }

    fn log_prob(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY // Exponential is only defined for x ≥ 0
        } else {
            self.rate.ln() - self.rate * x
        }
    }
}

/// Inverse-Gamma distribution with shape α and scale β parameters.
///
/// The inverse-gamma distribution is a continuous probability distribution commonly
/// used as a conjugate prior for the variance parameter of a normal distribution.
/// It is particularly useful in Bayesian statistics for modeling scale parameters.
///
/// # Mathematical Definition
/// The probability density function is:
/// ```text
/// f(x) = (β^α / Γ(α)) * x^(-α-1) * exp(-β/x) for x > 0
/// ```
/// where Γ is the gamma function.
///
/// # Common Usage
/// - Prior for variance parameters: σ² ~ InverseGamma(α, β)
/// - Prior for precision parameters: τ = 1/σ² ~ Gamma(α, β)
///
/// # Examples
/// ```
/// use ppf::distributions::{Distribution, InverseGamma};
/// use rand::rng;
///
/// // Inverse-Gamma with shape=2, scale=1 (mean=1, if α>1)
/// let inv_gamma = InverseGamma::new(2.0, 1.0);
///
/// // Sample from the distribution
/// let mut rng = rng();
/// let sample = inv_gamma.sample(&mut rng);
///
/// // Evaluate log probability (only positive values allowed)
/// let log_prob = inv_gamma.log_prob(1.5);
/// ```
#[derive(Debug, Clone)]
pub struct InverseGamma {
    /// Shape parameter (α) of the distribution
    shape: f64,
    /// Scale parameter (β) of the distribution
    scale: f64,
}

impl InverseGamma {
    /// Create a new inverse-gamma distribution with given shape and scale parameters.
    ///
    /// # Arguments
    /// * `shape` - The shape parameter (α), must be positive
    /// * `scale` - The scale parameter (β), must be positive
    ///
    /// # Panics
    /// This function will panic if either parameter is non-positive.
    ///
    /// # Examples
    /// ```
    /// use ppf::distributions::InverseGamma;
    ///
    /// // Weak prior for variance: α=1, β=1
    /// let weak_prior = InverseGamma::new(1.0, 1.0);
    ///
    /// // More informative: α=2, β=1
    /// let informative = InverseGamma::new(2.0, 1.0);
    /// ```
    pub fn new(shape: f64, scale: f64) -> Self {
        assert!(
            shape > 0.0,
            "Shape parameter must be positive, got {}",
            shape
        );
        assert!(
            scale > 0.0,
            "Scale parameter must be positive, got {}",
            scale
        );
        InverseGamma { shape, scale }
    }

    /// Get the shape parameter of the distribution.
    ///
    /// # Returns
    /// The shape parameter (α) of the inverse-gamma distribution
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the scale parameter of the distribution.
    ///
    /// # Returns
    /// The scale parameter (β) of the inverse-gamma distribution
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the mean of the distribution.
    ///
    /// # Returns
    /// The mean (β/(α-1)) if α > 1, otherwise undefined (returns NaN)
    pub fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            self.scale / (self.shape - 1.0)
        } else {
            f64::NAN // Mean undefined for α ≤ 1
        }
    }

    /// Get the variance of the distribution.
    ///
    /// # Returns
    /// The variance if α > 2, otherwise undefined (returns NaN)
    pub fn variance(&self) -> f64 {
        if self.shape > 2.0 {
            let alpha_minus_1 = self.shape - 1.0;
            (self.scale * self.scale) / (alpha_minus_1 * alpha_minus_1 * (self.shape - 2.0))
        } else {
            f64::NAN // Variance undefined for α ≤ 2
        }
    }
}

impl Distribution for InverseGamma {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Sample from Gamma(α, 1/β) and take reciprocal
        // Use acceptance-rejection method for gamma sampling
        let gamma_sample = sample_gamma(self.shape, 1.0 / self.scale, rng);
        1.0 / gamma_sample
    }

    fn log_prob(&self, x: f64) -> f64 {
        if x <= 0.0 {
            f64::NEG_INFINITY // Inverse-gamma is only defined for x > 0
        } else {
            // log(β^α / Γ(α)) - (α+1)*log(x) - β/x
            let log_beta_alpha = self.shape * self.scale.ln();
            let log_gamma_alpha = log_gamma_function(self.shape);
            let log_normalizing_constant = log_beta_alpha - log_gamma_alpha;

            log_normalizing_constant - (self.shape + 1.0) * x.ln() - self.scale / x
        }
    }
}

/// Sample from Gamma(shape, scale) distribution using Marsaglia and Tsang's method.
fn sample_gamma<R: Rng>(shape: f64, scale: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        // For α < 1, use transformation: if Y ~ Gamma(α+1), then X = Y * U^(1/α) ~ Gamma(α)
        let u: f64 = rng.random();
        sample_gamma(shape + 1.0, scale, rng) * u.powf(1.0 / shape)
    } else if shape == 1.0 {
        // Gamma(1, β) = Exponential(1/β)
        let u: f64 = rng.random();
        -u.ln() * scale
    } else {
        // Marsaglia and Tsang's method for α ≥ 1
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let mut x;
            let mut v;
            loop {
                // Sample from standard normal (Box-Muller transform)
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }

            v = v * v * v;
            let u: f64 = rng.random();

            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v * scale;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

/// Compute log of gamma function using a more stable approach.
/// This avoids deep recursion that can cause stack overflow.
fn log_gamma_function(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    // For small positive values, use the approximation directly
    if x < 0.5 {
        // Use Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-15 {
            return f64::INFINITY; // Near poles
        }
        return std::f64::consts::PI.ln() - sin_pi_x.abs().ln() - log_gamma_function(1.0 - x);
    }

    // Shift to x >= 1.5 to avoid recursion issues
    let mut y = x;
    let mut result = 0.0;

    while y < 1.5 {
        result -= y.ln();
        y += 1.0;
    }

    // Now use Stirling's approximation for y >= 1.5
    if y < 7.0 {
        // More accurate coefficients for smaller values
        let z = 1.0 / (y * y);
        let series = 1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0)));
        result + (y - 0.5) * y.ln() - y + 0.5 * (2.0 * std::f64::consts::PI).ln() + series / y
    } else {
        // Standard Stirling's approximation
        result + (y - 0.5) * y.ln() - y + 0.5 * (2.0 * std::f64::consts::PI).ln()
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

    #[test]
    fn test_exponential_properties() {
        let exp = Exponential::new(2.0);

        // Test basic properties
        assert_eq!(exp.rate(), 2.0);
        assert_eq!(exp.mean(), 0.5); // 1/λ = 1/2
        assert_eq!(exp.variance(), 0.25); // 1/λ² = 1/4

        // Test log prob
        assert_eq!(exp.log_prob(-1.0), f64::NEG_INFINITY); // Negative values
        assert!(exp.log_prob(0.0).is_finite());
        assert!(exp.log_prob(1.0) < 0.0); // Log prob should be negative
    }

    #[test]
    fn test_exponential_sampling() {
        let exp = Exponential::new(1.0);
        let mut rng = rng();

        // Generate samples
        let n = 20_000;
        let samples: Vec<f64> = (0..n).map(|_| exp.sample(&mut rng)).collect();

        // All samples should be positive
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Sample mean should be close to theoretical mean (1.0)
        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        assert!(
            (sample_mean - 1.0).abs() < 0.1,
            "Sample mean {:.3} should be close to 1.0",
            sample_mean
        );
    }

    #[test]
    fn test_inverse_gamma_properties() {
        let inv_gamma = InverseGamma::new(3.0, 2.0);

        // Test basic properties
        assert_eq!(inv_gamma.shape(), 3.0);
        assert_eq!(inv_gamma.scale(), 2.0);
        assert_eq!(inv_gamma.mean(), 1.0); // β/(α-1) = 2/(3-1) = 1
        assert!(inv_gamma.variance().is_finite()); // Should be defined for α > 2

        // Test boundary case where variance is undefined
        let inv_gamma_boundary = InverseGamma::new(2.0, 1.0);
        assert!(inv_gamma_boundary.variance().is_nan()); // Undefined for α ≤ 2

        // Test undefined cases
        let inv_gamma_small = InverseGamma::new(0.5, 1.0);
        assert!(inv_gamma_small.mean().is_nan()); // Undefined for α ≤ 1
    }

    #[test]
    fn test_inverse_gamma_log_prob() {
        let inv_gamma = InverseGamma::new(2.0, 1.0);

        // Test log prob properties
        assert_eq!(inv_gamma.log_prob(-1.0), f64::NEG_INFINITY); // Negative values
        assert_eq!(inv_gamma.log_prob(0.0), f64::NEG_INFINITY); // Zero
        assert!(inv_gamma.log_prob(1.0) < 0.0); // Positive values should give finite log prob
        assert!(inv_gamma.log_prob(0.1).is_finite());
    }

    #[test]
    fn test_inverse_gamma_sampling() {
        let inv_gamma = InverseGamma::new(3.0, 2.0);
        let mut rng = rng();

        // Generate samples
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| inv_gamma.sample(&mut rng)).collect();

        // All samples should be positive
        assert!(samples.iter().all(|&x| x > 0.0));

        // Sample mean should be reasonably close to theoretical mean
        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        let theoretical_mean = inv_gamma.mean(); // 2/(3-1) = 1.0
        assert!(
            (sample_mean - theoretical_mean).abs() < 0.2,
            "Sample mean {:.3} should be close to theoretical mean {:.3}",
            sample_mean,
            theoretical_mean
        );
    }

    #[test]
    #[should_panic]
    fn test_exponential_invalid_rate() {
        Exponential::new(-1.0); // Should panic
    }

    #[test]
    #[should_panic]
    fn test_inverse_gamma_invalid_shape() {
        InverseGamma::new(-1.0, 1.0); // Should panic
    }

    #[test]
    #[should_panic]
    fn test_inverse_gamma_invalid_scale() {
        InverseGamma::new(1.0, -1.0); // Should panic
    }
}
