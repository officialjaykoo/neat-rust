use std::error::Error;

use neat_rust::{
    algorithm::{
        Engine, EpochStopReason, EvolutionEngine, FitnessError, GenerationStats, GenomeId,
        GenomeProblem, Population, ProblemEvaluator,
    },
    io::Config,
    network::{GenomeCodec, NetworkCodec},
    DefaultGenome,
};

struct XorProblem {
    codec: NetworkCodec,
}

impl GenomeProblem for XorProblem {
    type Error = FitnessError;

    fn evaluate(
        &mut self,
        _genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, Self::Error> {
        let mut network = self
            .codec
            .decode(genome, config)
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

#[test]
fn evolution_engine_runs_problem_through_codec() -> Result<(), Box<dyn Error>> {
    let config = Config::from_toml_str(include_str!("../examples/xor_config.toml"))?;
    let population_size = config.neat.pop_size;
    let population = Population::new(config, 42)?;
    let evaluator = ProblemEvaluator::new(XorProblem {
        codec: NetworkCodec::from_config(),
    });
    let mut engine = EvolutionEngine::new(population, evaluator).with_generation_limit(1);

    let epoch = engine
        .next_epoch()?
        .expect("generation limit should still allow one epoch");
    let stats: GenerationStats = epoch.stats.expect("epoch should include generation stats");

    assert_eq!(epoch.stop_reason, Some(EpochStopReason::GenerationLimit));
    assert_eq!(stats.generation, 0);
    assert_eq!(stats.population_size, population_size);
    assert_eq!(stats.evaluated_count, population_size);
    assert!(stats.best_fitness.is_finite());
    assert!(engine.next_epoch()?.is_none());
    Ok(())
}
