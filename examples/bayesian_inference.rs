use ppf::distributions::*;
use ppf::model::Model;
use ppf::samplers::*;

fn main() {
    // Data: observed values
    let data = [5.0, 7.0, 4.0, 6.0];

    // Calculate data mean before moving data into closure
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
    let samples = mh.sample(&model, 10_000);

    println!("Posterior mean estimate for mu = {:.3}", samples.mean(0));
    println!("Posterior std estimate for mu = {:.3}", samples.std(0));
    println!(
        "95% credible interval: [{:.3}, {:.3}]",
        samples.quantile(0, 0.025),
        samples.quantile(0, 0.975)
    );

    // Compare with data
    println!("Data mean: {:.3}", data_mean);
}
