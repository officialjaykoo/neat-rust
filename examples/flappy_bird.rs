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

const WIDTH: f64 = 600.0;
const HEIGHT: f64 = 600.0;
const BIRD_X: f64 = 90.0;
const BIRD_RADIUS: f64 = 15.0;
const PIPE_WIDTH: f64 = 62.0;
const PIPE_GAP: f64 = 94.0;
const PIPE_SPEED: f64 = 4.8;
const PIPE_SPACING: f64 = 132.0;
const GRAVITY: f64 = 0.60;
const FLAP_VELOCITY: f64 = -8.2;
const MAX_STEPS: usize = 3000;
const TRAIN_EPISODES: [u64; 8] = [17, 31, 47, 73, 101, 137, 173, 197];
const REPORT_EPISODES: [u64; 10] = [131, 157, 181, 211, 241, 277, 307, 331, 367, 397];

#[derive(Debug, Clone, Copy)]
struct Pipe {
    x: f64,
    gap_center: f64,
    scored: bool,
}

#[derive(Debug, Clone)]
struct FlappyGame {
    y: f64,
    velocity: f64,
    pipes: Vec<Pipe>,
    rng: XorShiftRng,
    steps: usize,
    passed: usize,
    fitness: f64,
}

#[derive(Debug, Clone, Copy)]
struct EpisodeReport {
    fitness: f64,
    passed: usize,
    steps: usize,
}

struct FlappyEvaluator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlappyConfigProfile {
    Plain,
    NodeGru,
    Hebbian,
    LinearGate,
    RgLruLite,
}

impl FlappyConfigProfile {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("plain").to_ascii_lowercase().as_str() {
            "plain" | "base" | "no-gru" | "no_gru" => Ok(Self::Plain),
            "gru" | "node-gru" | "node_gru" | "nodegru" => Ok(Self::NodeGru),
            "hebbian" | "node-hebbian" | "node_hebbian" => Ok(Self::Hebbian),
            "linear" | "linear-gate" | "linear_gate" | "node-linear-gate"
            | "node_linear_gate" => Ok(Self::LinearGate),
            "rg-lru-lite" | "rg_lru_lite" | "linear-gate-v2" | "linear_gate_v2" => {
                Ok(Self::RgLruLite)
            }
            other => Err(format!(
                "unknown flappy config profile {other:?}; use plain, node-gru, hebbian, linear-gate, or rg-lru-lite"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::NodeGru => "node-gru",
            Self::Hebbian => "hebbian",
            Self::LinearGate => "linear-gate",
            Self::RgLruLite => "rg-lru-lite",
        }
    }

    fn config_text(self) -> &'static str {
        match self {
            Self::Plain => include_str!("flappy_bird_plain_config.toml"),
            Self::NodeGru => include_str!("flappy_bird_node_gru_config.toml"),
            Self::Hebbian => include_str!("flappy_bird_hebbian_config.toml"),
            Self::LinearGate => include_str!("flappy_bird_linear_gate_config.toml"),
            Self::RgLruLite => include_str!("flappy_bird_rg_lru_lite_config.toml"),
        }
    }
}

impl FitnessEvaluator for FlappyEvaluator {
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

impl FlappyGame {
    fn new(seed: u64) -> Self {
        let mut game = Self {
            y: HEIGHT * 0.5,
            velocity: 0.0,
            pipes: Vec::new(),
            rng: XorShiftRng::seed_from_u64(seed),
            steps: 0,
            passed: 0,
            fitness: 0.0,
        };
        game.push_pipe(WIDTH + 80.0);
        game.push_pipe(WIDTH + 80.0 + PIPE_SPACING);
        game
    }

    fn inputs(&self) -> [f64; 6] {
        let pipe = self.next_pipe();
        let gap_top = pipe.gap_center - (PIPE_GAP * 0.5);
        let gap_bottom = pipe.gap_center + (PIPE_GAP * 0.5);
        [
            ((self.y / HEIGHT) * 2.0) - 1.0,
            (self.velocity / 10.0).clamp(-1.0, 1.0),
            ((pipe.x - BIRD_X) / WIDTH).clamp(-1.0, 1.0),
            ((pipe.gap_center - self.y) / (HEIGHT * 0.5)).clamp(-1.0, 1.0),
            ((gap_top - self.y) / (HEIGHT * 0.5)).clamp(-1.0, 1.0),
            ((gap_bottom - self.y) / (HEIGHT * 0.5)).clamp(-1.0, 1.0),
        ]
    }

    fn step(&mut self, jump: bool) -> bool {
        let previous_gap_error = (self.next_pipe().gap_center - self.y).abs();
        if jump {
            self.velocity = FLAP_VELOCITY;
            self.fitness -= 0.03;
        }

        self.velocity = (self.velocity + GRAVITY).clamp(-9.0, 10.5);
        self.y += self.velocity;
        for pipe in &mut self.pipes {
            pipe.x -= PIPE_SPEED;
        }

        if self
            .pipes
            .last()
            .map(|pipe| pipe.x < WIDTH - PIPE_SPACING)
            .unwrap_or(true)
        {
            self.push_pipe(WIDTH + PIPE_WIDTH);
        }
        self.pipes.retain(|pipe| pipe.x + PIPE_WIDTH > -20.0);

        for pipe in &mut self.pipes {
            if !pipe.scored && pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS {
                pipe.scored = true;
                self.passed += 1;
                self.fitness += 90.0 + (self.passed as f64 * 8.0);
            }
        }

        self.steps += 1;
        self.fitness += 0.12;
        let next_gap_error = (self.next_pipe().gap_center - self.y).abs();
        if next_gap_error < previous_gap_error {
            self.fitness += 0.10;
        } else {
            self.fitness -= 0.02;
        }

        self.steps < MAX_STEPS && !self.collided()
    }

    fn collided(&self) -> bool {
        if self.y - BIRD_RADIUS <= 0.0 || self.y + BIRD_RADIUS >= HEIGHT {
            return true;
        }
        self.pipes.iter().any(|pipe| {
            let overlaps_x =
                BIRD_X + BIRD_RADIUS >= pipe.x && BIRD_X - BIRD_RADIUS <= pipe.x + PIPE_WIDTH;
            let gap_top = pipe.gap_center - (PIPE_GAP * 0.5);
            let gap_bottom = pipe.gap_center + (PIPE_GAP * 0.5);
            overlaps_x && (self.y - BIRD_RADIUS < gap_top || self.y + BIRD_RADIUS > gap_bottom)
        })
    }

    fn next_pipe(&self) -> Pipe {
        self.pipes
            .iter()
            .copied()
            .find(|pipe| pipe.x + PIPE_WIDTH >= BIRD_X - BIRD_RADIUS)
            .or_else(|| self.pipes.last().copied())
            .unwrap_or(Pipe {
                x: WIDTH,
                gap_center: HEIGHT * 0.5,
                scored: false,
            })
    }

    fn push_pipe(&mut self, x: f64) {
        let margin = 58.0;
        let min_center = margin + (PIPE_GAP * 0.5);
        let max_center = HEIGHT - margin - (PIPE_GAP * 0.5);
        let mut gap_center = min_center + ((max_center - min_center) * self.rng.next_f64());
        if let Some(previous) = self.pipes.last() {
            let max_shift = 165.0;
            gap_center = gap_center.clamp(
                (previous.gap_center - max_shift).max(min_center),
                (previous.gap_center + max_shift).min(max_center),
            );
        }
        self.pipes.push(Pipe {
            x,
            gap_center,
            scored: false,
        });
    }
}

fn run_episode(seed: u64, network: &mut RecurrentNetwork) -> Result<EpisodeReport, FitnessError> {
    network.reset();
    let mut game = FlappyGame::new(seed);
    while game.step(
        network
            .activate(&game.inputs())
            .map_err(|err| FitnessError::new(err.to_string()))?
            .first()
            .copied()
            .unwrap_or(0.0)
            > 0.0,
    ) {}
    Ok(EpisodeReport {
        fitness: game.fitness.max(0.0),
        passed: game.passed,
        steps: game.steps,
    })
}

fn evaluate_report(genome: &DefaultGenome, config: &Config) -> Result<EpisodeReport, FitnessError> {
    let mut total = EpisodeReport {
        fitness: 0.0,
        passed: 0,
        steps: 0,
    };
    for seed in REPORT_EPISODES {
        let mut network = RecurrentNetwork::create(genome, &config.genome)
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let report = run_episode(seed, &mut network)?;
        total.fitness += report.fitness;
        total.passed += report.passed;
        total.steps += report.steps;
    }
    Ok(EpisodeReport {
        fitness: total.fitness / REPORT_EPISODES.len() as f64,
        passed: total.passed,
        steps: total.steps / REPORT_EPISODES.len(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let generations = args
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12);
    let bootstrap_rounds = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let profile = FlappyConfigProfile::parse(args.get(2).map(String::as_str))?;
    let base_seed = args
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(404);
    let config = Config::from_toml_str(profile.config_text())?;
    let mut evaluator = FlappyEvaluator;
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
    evaluator: &mut FlappyEvaluator,
    generations: usize,
) -> Result<DefaultGenome, Box<dyn Error>> {
    Ok(population
        .run_with_evaluator(evaluator, Some(generations))?
        .expect("population should keep a champion"))
}

fn print_report(
    profile: FlappyConfigProfile,
    label: &str,
    generations: usize,
    genome: &DefaultGenome,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    let report = evaluate_report(genome, config)?;
    println!(
        "flappy_bird profile={} {label} generations={generations} best_genome={} train_fitness={:.2} report_fitness={:.2} report_pipes={} avg_steps={}",
        profile.name(),
        genome.key,
        genome.fitness.unwrap_or(0.0),
        report.fitness,
        report.passed,
        report.steps
    );
    Ok(())
}
