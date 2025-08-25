use ppf::distributions::*;
use rand::rng;

fn main() {
    let normal = Normal::new(0.0, 1.0);
    let mut rng = rng();

    for _ in 0..5 {
        let x = normal.sample(&mut rng);
        println!("Sample: {:.4}, log_prob: {:.4}", x, normal.log_prob(x));
    }
}
