use std::error::Error;

use neat_rust::{
    algorithm::{DefaultGenome, FitnessError, FitnessEvaluator, GenomeId, Population},
    io::Config,
    network::FeedForwardNetwork,
};

struct XorEvaluator;

impl FitnessEvaluator for XorEvaluator {
    fn evaluate_genome(
        &mut self,
        _genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let mut network = FeedForwardNetwork::create(genome, &config.genome)
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let cases = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];
        let error = cases
            .into_iter()
            .map(|(inputs, expected)| {
                let output = network
                    .activate(&inputs)
                    .map(|values| values[0])
                    .unwrap_or(0.0);
                (expected - output).abs()
            })
            .sum::<f64>();
        Ok(4.0 - error)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_toml_str(include_str!("xor_config.toml"))?;
    let mut population = Population::new(config, 100)?;
    let mut evaluator = XorEvaluator;
    let best = population
        .run_with_evaluator(&mut evaluator, Some(20))?
        .expect("population should keep a champion");

    println!(
        "best_genome={} fitness={:.4}",
        best.key,
        best.fitness.unwrap_or(0.0)
    );
    Ok(())
}
