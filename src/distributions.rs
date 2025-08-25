use rand::Rng;
use rand_distr::Distribution as RandDistribution;


pub trait Distribution {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64;
    fn log_prob(&self, x: f64) -> f64;
}

pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Normal { mu, sigma }
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
