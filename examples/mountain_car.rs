use std::env;
use std::error::Error;

use neat_rust::{
    algorithm::{
        BootstrapStrategy, DefaultGenome, FitnessError, FitnessEvaluator, GenomeId, Population,
        RandomSource, XorShiftRng,
    },
    io::Config,
    network::RecurrentNetwork,
};

const MIN_POSITION: f64 = -1.2;
const MAX_POSITION: f64 = 0.6;
const MAX_SPEED: f64 = 0.07;
const GOAL_POSITION: f64 = 0.5;
const FORCE: f64 = 0.001;
const GRAVITY: f64 = 0.0025;
const MAX_STEPS: usize = 200;
const TRAIN_EPISODES: [u64; 12] = [11, 17, 23, 31, 43, 59, 71, 83, 97, 109, 127, 139];
const REPORT_EPISODES: [u64; 16] = [
    151, 167, 181, 193, 211, 227, 241, 257, 277, 293, 311, 331, 347, 367, 383, 397,
];

#[derive(Debug, Clone, Copy)]
struct MountainCarState {
    position: f64,
    velocity: f64,
}

#[derive(Debug, Clone, Copy)]
enum MountainCarAction {
    Left,
    Neutral,
    Right,
}

#[derive(Debug, Clone, Copy)]
struct EpisodeReport {
    fitness: f64,
    steps: usize,
    solved: bool,
    max_position: f64,
}

struct MountainCarEvaluator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MountainCarConfigProfile {
    Plain,
    NodeGru,
    Hebbian,
}

impl MountainCarConfigProfile {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("plain").to_ascii_lowercase().as_str() {
            "plain" | "base" | "no-gru" | "no_gru" => Ok(Self::Plain),
            "gru" | "node-gru" | "node_gru" | "nodegru" => Ok(Self::NodeGru),
            "hebbian" | "node-hebbian" | "node_hebbian" => Ok(Self::Hebbian),
            other => Err(format!(
                "unknown mountain car config profile {other:?}; use plain, node-gru, or hebbian"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::NodeGru => "node-gru",
            Self::Hebbian => "hebbian",
        }
    }

    fn config_text(self) -> &'static str {
        match self {
            Self::Plain => include_str!("mountain_car_plain_config.toml"),
            Self::NodeGru => include_str!("mountain_car_node_gru_config.toml"),
            Self::Hebbian => include_str!("mountain_car_hebbian_config.toml"),
        }
    }
}

impl FitnessEvaluator for MountainCarEvaluator {
    fn evaluate_genome(
        &mut self,
        _genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let mut total = 0.0;
        for seed in TRAIN_EPISODES {
            let mut network = RecurrentNetwork::create(genome, &config.genome)
                .map_err(|err| FitnessError::new(err.to_string()))?;
            total += run_episode(seed, &mut network)?.fitness;
        }
        Ok(total / TRAIN_EPISODES.len() as f64)
    }
}

impl MountainCarState {
    fn new(seed: u64) -> Self {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        Self {
            position: -0.6 + (0.2 * rng.next_f64()),
            velocity: 0.0,
        }
    }

    fn inputs(self) -> [f64; 4] {
        [
            scale(self.position, MIN_POSITION, MAX_POSITION),
            scale(
                GOAL_POSITION - self.position,
                0.0,
                GOAL_POSITION - MIN_POSITION,
            ),
            (3.0 * self.position).cos(),
            1.0,
        ]
    }

    fn step(&mut self, action: MountainCarAction) {
        let force = match action {
            MountainCarAction::Left => -FORCE,
            MountainCarAction::Neutral => 0.0,
            MountainCarAction::Right => FORCE,
        };
        self.velocity += force - (GRAVITY * (3.0 * self.position).cos());
        self.velocity = self.velocity.clamp(-MAX_SPEED, MAX_SPEED);
        self.position += self.velocity;
        self.position = self.position.clamp(MIN_POSITION, MAX_POSITION);
        if self.position <= MIN_POSITION && self.velocity < 0.0 {
            self.velocity = 0.0;
        }
    }
}

impl MountainCarAction {
    fn from_outputs(outputs: &[f64]) -> Self {
        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, value) in outputs.iter().copied().enumerate().take(3) {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        match best_index {
            0 => Self::Left,
            1 => Self::Neutral,
            _ => Self::Right,
        }
    }
}

fn run_episode(seed: u64, network: &mut RecurrentNetwork) -> Result<EpisodeReport, FitnessError> {
    network.reset();
    let mut state = MountainCarState::new(seed);
    let mut max_position = state.position;
    let mut energy_gain = 0.0;
    let mut last_energy = car_energy(state);

    for step in 0..MAX_STEPS {
        let outputs = network
            .activate(&state.inputs())
            .map_err(|err| FitnessError::new(err.to_string()))?;
        state.step(MountainCarAction::from_outputs(&outputs));
        max_position = max_position.max(state.position);

        let energy = car_energy(state);
        energy_gain += (energy - last_energy).max(-0.005);
        last_energy = energy;

        if state.position >= GOAL_POSITION {
            return Ok(EpisodeReport {
                fitness: 1000.0 + ((MAX_STEPS - step) as f64 * 2.0) + (max_position * 100.0),
                steps: step + 1,
                solved: true,
                max_position,
            });
        }
    }

    let progress = scale(max_position, MIN_POSITION, GOAL_POSITION).max(0.0);
    let velocity_bonus = state.velocity.abs() * 50.0;
    Ok(EpisodeReport {
        fitness: (progress * 500.0) + (energy_gain * 200.0) + velocity_bonus,
        steps: MAX_STEPS,
        solved: false,
        max_position,
    })
}

fn evaluate_report(genome: &DefaultGenome, config: &Config) -> Result<EpisodeReport, FitnessError> {
    let mut total = EpisodeReport {
        fitness: 0.0,
        steps: 0,
        solved: false,
        max_position: f64::NEG_INFINITY,
    };
    let mut solved = 0usize;
    for seed in REPORT_EPISODES {
        let mut network = RecurrentNetwork::create(genome, &config.genome)
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let report = run_episode(seed, &mut network)?;
        total.fitness += report.fitness;
        total.steps += report.steps;
        total.max_position = total.max_position.max(report.max_position);
        solved += usize::from(report.solved);
    }
    Ok(EpisodeReport {
        fitness: total.fitness / REPORT_EPISODES.len() as f64,
        steps: total.steps / REPORT_EPISODES.len(),
        solved: solved == REPORT_EPISODES.len(),
        max_position: total.max_position,
    })
}

fn car_energy(state: MountainCarState) -> f64 {
    (0.5 * state.velocity * state.velocity) + (0.0025 * (3.0 * state.position).sin())
}

fn scale(value: f64, min: f64, max: f64) -> f64 {
    (((value - min) / (max - min)) * 2.0 - 1.0).clamp(-1.0, 1.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let generations = args
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50);
    let bootstrap_rounds = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let profile = MountainCarConfigProfile::parse(args.get(2).map(String::as_str))?;
    let base_seed = args
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(606);
    let config = Config::from_toml_str(profile.config_text())?;
    let mut evaluator = MountainCarEvaluator;
    let mut best = run_population(
        Population::new(config.clone(), base_seed)?,
        &mut evaluator,
        generations,
    )?;
    print_report(
        profile,
        "round=0 bootstrap=false",
        generations,
        &best,
        &config,
    )?;

    for round in 1..=bootstrap_rounds {
        let strategy = BootstrapStrategy::from_champion(best.clone(), 0.5);
        let candidate = run_population(
            Population::new_with_bootstrap(config.clone(), base_seed + round as u64, strategy)?,
            &mut evaluator,
            generations,
        )?;
        print_report(
            profile,
            &format!("round={round} bootstrap=true"),
            generations,
            &candidate,
            &config,
        )?;
        if candidate.fitness.unwrap_or(f64::NEG_INFINITY)
            > best.fitness.unwrap_or(f64::NEG_INFINITY)
        {
            best = candidate;
        }
    }

    print_report(profile, "final", generations, &best, &config)?;
    Ok(())
}

fn run_population(
    mut population: Population,
    evaluator: &mut MountainCarEvaluator,
    generations: usize,
) -> Result<DefaultGenome, Box<dyn Error>> {
    Ok(population
        .run_with_evaluator(evaluator, Some(generations))?
        .expect("population should keep a champion"))
}

fn print_report(
    profile: MountainCarConfigProfile,
    label: &str,
    generations: usize,
    genome: &DefaultGenome,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    let report = evaluate_report(genome, config)?;
    println!(
        "mountain_car profile={} {label} generations={generations} best_genome={} train_fitness={:.2} report_fitness={:.2} report_solved={} avg_steps={} max_position={:.3}",
        profile.name(),
        genome.key,
        genome.fitness.unwrap_or(0.0),
        report.fitness,
        report.solved,
        report.steps,
        report.max_position
    );
    Ok(())
}
