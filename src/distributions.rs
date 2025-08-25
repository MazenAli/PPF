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
