//! Sample collection and statistical analysis for MCMC output.
//!
//! This module provides the [`Samples`] struct for storing and analyzing
//! collections of parameter samples from MCMC runs.

/// Container for MCMC samples with comprehensive statistical analysis methods.
///
/// The `Samples` struct stores a collection of parameter vectors from MCMC sampling
/// and provides methods for computing posterior statistics like means, variances,
/// quantiles, and credible intervals.
///
/// # Structure
///
/// Samples are stored as a matrix where:
/// - Each row represents one MCMC sample (iteration)
/// - Each column represents one parameter
/// - All samples must have the same number of parameters
///
/// # Examples
///
/// ```
/// use ppf::samples::Samples;
///
/// // Create container for 2 parameters
/// let mut samples = Samples::new(2);
///
/// // Add some samples
/// samples.push(vec![1.0, 2.0]);
/// samples.push(vec![1.1, 2.1]);
/// samples.push(vec![0.9, 1.9]);
///
/// // Compute statistics
/// let mean_param0 = samples.mean(0);
/// let std_param1 = samples.std(1);
/// let median = samples.quantile(0, 0.5);
/// ```
pub struct Samples {
    /// Matrix of samples where each row is a sample and each column is a parameter
    samples: Vec<Vec<f64>>,
    /// Number of parameters
    n_params: usize,
}

impl Samples {
    /// Create a new Samples container.
    ///
    /// # Arguments
    /// * `n_params` - Number of parameters in the model
    pub fn new(n_params: usize) -> Self {
        Samples {
            samples: Vec::new(),
            n_params,
        }
    }

    /// Add a sample to the collection.
    ///
    /// # Arguments
    /// * `sample` - Parameter values for this sample
    ///
    /// # Panics
    /// Panics if the sample length doesn't match the expected number of parameters
    pub fn push(&mut self, sample: Vec<f64>) {
        assert_eq!(
            sample.len(),
            self.n_params,
            "Sample length {} doesn't match expected parameters {}",
            sample.len(),
            self.n_params
        );
        self.samples.push(sample);
    }

    /// Get the number of samples collected.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the samples collection is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Calculate the sample mean for a specific parameter.
    ///
    /// # Arguments
    /// * `param_idx` - Index of the parameter (0-based)
    ///
    /// # Returns
    /// Sample mean of the specified parameter
    ///
    /// # Panics
    /// Panics if param_idx is out of bounds or if there are no samples
    pub fn mean(&self, param_idx: usize) -> f64 {
        assert!(
            param_idx < self.n_params,
            "Parameter index {} out of bounds",
            param_idx
        );
        assert!(!self.is_empty(), "Cannot compute mean of empty samples");

        let sum: f64 = self.samples.iter().map(|sample| sample[param_idx]).sum();
        sum / self.len() as f64
    }

    /// Calculate the sample variance for a specific parameter.
    ///
    /// # Arguments
    /// * `param_idx` - Index of the parameter (0-based)
    ///
    /// # Returns
    /// Sample variance of the specified parameter
    pub fn var(&self, param_idx: usize) -> f64 {
        assert!(
            param_idx < self.n_params,
            "Parameter index {} out of bounds",
            param_idx
        );
        assert!(
            self.len() > 1,
            "Need at least 2 samples to compute variance"
        );

        let mean = self.mean(param_idx);
        let sum_sq_diff: f64 = self
            .samples
            .iter()
            .map(|sample| {
                let diff = sample[param_idx] - mean;
                diff * diff
            })
            .sum();
        sum_sq_diff / (self.len() - 1) as f64
    }

    /// Calculate the sample standard deviation for a specific parameter.
    ///
    /// # Arguments  
    /// * `param_idx` - Index of the parameter (0-based)
    ///
    /// # Returns
    /// Sample standard deviation of the specified parameter
    pub fn std(&self, param_idx: usize) -> f64 {
        self.var(param_idx).sqrt()
    }

    /// Get a specific sample.
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample (0-based)
    ///
    /// # Returns
    /// Reference to the sample parameters
    ///
    /// # Panics
    /// Panics if sample_idx is out of bounds
    pub fn get(&self, sample_idx: usize) -> &[f64] {
        &self.samples[sample_idx]
    }

    /// Get all samples for a specific parameter.
    ///
    /// # Arguments
    /// * `param_idx` - Index of the parameter (0-based)
    ///
    /// # Returns
    /// Vector containing all samples for the specified parameter
    pub fn get_param(&self, param_idx: usize) -> Vec<f64> {
        assert!(
            param_idx < self.n_params,
            "Parameter index {} out of bounds",
            param_idx
        );

        self.samples
            .iter()
            .map(|sample| sample[param_idx])
            .collect()
    }

    /// Calculate a quantile for a specific parameter.
    ///
    /// # Arguments
    /// * `param_idx` - Index of the parameter (0-based)
    /// * `q` - Quantile to calculate (between 0 and 1)
    ///
    /// # Returns
    /// The q-th quantile of the specified parameter
    pub fn quantile(&self, param_idx: usize, q: f64) -> f64 {
        assert!(
            param_idx < self.n_params,
            "Parameter index {} out of bounds",
            param_idx
        );
        assert!((0.0..=1.0).contains(&q), "Quantile must be between 0 and 1");
        assert!(!self.is_empty(), "Cannot compute quantile of empty samples");

        let mut values = self.get_param(param_idx);
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (q * (values.len() - 1) as f64).round() as usize;
        values[idx.min(values.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_creation() {
        let samples = Samples::new(2);
        assert_eq!(samples.n_params(), 2);
        assert_eq!(samples.len(), 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_adding_samples() {
        let mut samples = Samples::new(2);
        samples.push(vec![1.0, 2.0]);
        samples.push(vec![3.0, 4.0]);

        assert_eq!(samples.len(), 2);
        assert!(!samples.is_empty());
        assert_eq!(samples.get(0), &[1.0, 2.0]);
        assert_eq!(samples.get(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_mean_calculation() {
        let mut samples = Samples::new(2);
        samples.push(vec![1.0, 2.0]);
        samples.push(vec![3.0, 4.0]);
        samples.push(vec![5.0, 6.0]);

        assert_eq!(samples.mean(0), 3.0); // (1+3+5)/3
        assert_eq!(samples.mean(1), 4.0); // (2+4+6)/3
    }

    #[test]
    fn test_variance_calculation() {
        let mut samples = Samples::new(1);
        samples.push(vec![1.0]);
        samples.push(vec![2.0]);
        samples.push(vec![3.0]);

        // Variance should be 1.0 for [1,2,3]
        assert!((samples.var(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_param() {
        let mut samples = Samples::new(2);
        samples.push(vec![1.0, 10.0]);
        samples.push(vec![2.0, 20.0]);
        samples.push(vec![3.0, 30.0]);

        assert_eq!(samples.get_param(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(samples.get_param(1), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_quantile() {
        let mut samples = Samples::new(1);
        for i in 1..=100 {
            samples.push(vec![i as f64]);
        }

        assert_eq!(samples.quantile(0, 0.0), 1.0); // min
        // For 100 values, the median could be 50 or 51 depending on implementation
        let median = samples.quantile(0, 0.5);
        assert!(median >= 50.0 && median <= 51.0); // median
        assert_eq!(samples.quantile(0, 1.0), 100.0); // max
    }

    #[test]
    #[should_panic]
    fn test_wrong_sample_size() {
        let mut samples = Samples::new(2);
        samples.push(vec![1.0]); // Should panic - wrong size
    }
}
