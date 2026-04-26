use std::collections::VecDeque;
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

const GRID: i32 = 12;
const MAX_STEPS: usize = 180;
const STARVE_STEPS: usize = 64;
const TRAIN_EPISODES: [u64; 6] = [11, 23, 37, 53, 71, 97];
const REPORT_EPISODES: [u64; 8] = [131, 149, 167, 181, 199, 211, 229, 251];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Debug, Clone)]
struct SnakeGame {
    body: VecDeque<Point>,
    direction: Direction,
    apple: Point,
    rng: XorShiftRng,
    steps: usize,
    steps_since_apple: usize,
    eaten: usize,
    fitness: f64,
}

#[derive(Debug, Clone, Copy)]
struct EpisodeReport {
    fitness: f64,
    eaten: usize,
    steps: usize,
}

struct SnakeEvaluator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnakeConfigProfile {
    Plain,
    NodeGru,
    Hebbian,
    LinearGate,
    RgLruLite,
}

impl SnakeConfigProfile {
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
                "unknown snake config profile {other:?}; use plain, node-gru, hebbian, linear-gate, or rg-lru-lite"
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
            Self::Plain => include_str!("snake_cli_plain_config.toml"),
            Self::NodeGru => include_str!("snake_cli_node_gru_config.toml"),
            Self::Hebbian => include_str!("snake_cli_hebbian_config.toml"),
            Self::LinearGate => include_str!("snake_cli_linear_gate_config.toml"),
            Self::RgLruLite => include_str!("snake_cli_rg_lru_lite_config.toml"),
        }
    }
}

impl FitnessEvaluator for SnakeEvaluator {
    fn evaluate_genome(
        &mut self,
        genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let mut total = 0.0;
        for seed in TRAIN_EPISODES {
            let mut network = RecurrentNetwork::create(genome, &config.genome)
                .map_err(|err| FitnessError::new(err.to_string()))?;
            total += run_episode(genome_id, seed, &mut network)?.fitness;
        }
        Ok(total / TRAIN_EPISODES.len() as f64)
    }
}

impl SnakeGame {
    fn new(seed: u64) -> Self {
        let mut body = VecDeque::new();
        body.push_back(Point::new(5, 5));
        body.push_back(Point::new(4, 5));
        body.push_back(Point::new(3, 5));
        let mut game = Self {
            body,
            direction: Direction::Right,
            apple: Point::new(8, 5),
            rng: XorShiftRng::seed_from_u64(seed),
            steps: 0,
            steps_since_apple: 0,
            eaten: 0,
            fitness: 4.0,
        };
        game.apple = game.random_empty_cell();
        game
    }

    fn inputs(&self) -> [f64; 8] {
        let head = self.head();
        let apple_dx = (self.apple.x - head.x) as f64 / (GRID - 1) as f64;
        let apple_dy = (self.apple.y - head.y) as f64 / (GRID - 1) as f64;
        let forward = self.danger(self.direction);
        let left = self.danger(self.direction.left());
        let right = self.danger(self.direction.right());
        let (dir_x, dir_y) = self.direction.vector();
        [
            apple_dx,
            apple_dy,
            if forward { 1.0 } else { -1.0 },
            if left { 1.0 } else { -1.0 },
            if right { 1.0 } else { -1.0 },
            dir_x as f64,
            dir_y as f64,
            (self.body.len() as f64 / (GRID * GRID) as f64).min(1.0),
        ]
    }

    fn step(&mut self, chosen: Direction) -> bool {
        if chosen != self.direction.opposite() {
            self.direction = chosen;
        }

        let previous_distance = self.manhattan_to_apple();
        let next_head = self.head().shift(self.direction);
        let growing = next_head == self.apple;
        if self.collides(next_head, growing) {
            self.fitness -= 15.0;
            return false;
        }

        self.body.push_front(next_head);
        if growing {
            self.eaten += 1;
            self.steps_since_apple = 0;
            self.fitness += 40.0 + (self.eaten as f64 * 8.0);
            self.apple = self.random_empty_cell();
        } else {
            self.body.pop_back();
            self.steps_since_apple += 1;
        }

        let next_distance = self.manhattan_to_apple();
        self.steps += 1;
        self.fitness += 0.05;
        if next_distance < previous_distance {
            self.fitness += 0.40;
        } else {
            self.fitness -= 0.15;
        }

        self.steps < MAX_STEPS && self.steps_since_apple < STARVE_STEPS
    }

    fn head(&self) -> Point {
        self.body.front().copied().unwrap_or(Point::new(0, 0))
    }

    fn danger(&self, direction: Direction) -> bool {
        self.collides(self.head().shift(direction), false)
    }

    fn collides(&self, point: Point, growing: bool) -> bool {
        if point.x < 0 || point.y < 0 || point.x >= GRID || point.y >= GRID {
            return true;
        }
        let ignored_tail = if growing { 0 } else { 1 };
        let checked_len = self.body.len().saturating_sub(ignored_tail);
        self.body
            .iter()
            .take(checked_len)
            .any(|cell| *cell == point)
    }

    fn manhattan_to_apple(&self) -> i32 {
        (self.apple.x - self.head().x).abs() + (self.apple.y - self.head().y).abs()
    }

    fn random_empty_cell(&mut self) -> Point {
        let mut empty = Vec::new();
        for y in 0..GRID {
            for x in 0..GRID {
                let point = Point::new(x, y);
                if !self.body.iter().any(|cell| *cell == point) {
                    empty.push(point);
                }
            }
        }
        let Some(index) = self.rng.next_index(empty.len()) else {
            return self.head();
        };
        empty[index]
    }
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn shift(self, direction: Direction) -> Self {
        let (dx, dy) = direction.vector();
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

impl Direction {
    const ALL: [Direction; 4] = [
        Direction::Up,
        Direction::Right,
        Direction::Down,
        Direction::Left,
    ];

    fn from_outputs(outputs: &[f64]) -> Self {
        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, value) in outputs.iter().copied().enumerate().take(Self::ALL.len()) {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        Self::ALL[best_index]
    }

    fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Right => Self::Left,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
        }
    }

    fn left(self) -> Self {
        match self {
            Self::Up => Self::Left,
            Self::Right => Self::Up,
            Self::Down => Self::Right,
            Self::Left => Self::Down,
        }
    }

    fn right(self) -> Self {
        match self {
            Self::Up => Self::Right,
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
        }
    }

    fn vector(self) -> (i32, i32) {
        match self {
            Self::Up => (0, -1),
            Self::Right => (1, 0),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
        }
    }
}

fn run_episode(
    genome_id: GenomeId,
    seed: u64,
    network: &mut RecurrentNetwork,
) -> Result<EpisodeReport, FitnessError> {
    network.reset();
    let genome_seed = genome_id.raw().max(0) as u64;
    let mut game = SnakeGame::new(seed ^ (genome_seed << 32));
    while game.step(Direction::from_outputs(
        &network
            .activate(&game.inputs())
            .map_err(|err| FitnessError::new(err.to_string()))?,
    )) {}
    Ok(EpisodeReport {
        fitness: game.fitness.max(0.0),
        eaten: game.eaten,
        steps: game.steps,
    })
}

fn evaluate_report(genome: &DefaultGenome, config: &Config) -> Result<EpisodeReport, FitnessError> {
    let mut total = EpisodeReport {
        fitness: 0.0,
        eaten: 0,
        steps: 0,
    };
    for seed in REPORT_EPISODES {
        let mut network = RecurrentNetwork::create(genome, &config.genome)
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let report = run_episode(GenomeId::new(0), seed, &mut network)?;
        total.fitness += report.fitness;
        total.eaten += report.eaten;
        total.steps += report.steps;
    }
    Ok(EpisodeReport {
        fitness: total.fitness / REPORT_EPISODES.len() as f64,
        eaten: total.eaten,
        steps: total.steps / REPORT_EPISODES.len(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let generations = args
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let bootstrap_rounds = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let profile = SnakeConfigProfile::parse(args.get(2).map(String::as_str))?;
    let base_seed = args
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(303);
    let config = Config::from_toml_str(profile.config_text())?;
    let mut evaluator = SnakeEvaluator;
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
    evaluator: &mut SnakeEvaluator,
    generations: usize,
) -> Result<DefaultGenome, Box<dyn Error>> {
    Ok(population
        .run_with_evaluator(evaluator, Some(generations))?
        .expect("population should keep a champion"))
}

fn print_report(
    profile: SnakeConfigProfile,
    label: &str,
    generations: usize,
    genome: &DefaultGenome,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    let report = evaluate_report(genome, config)?;
    println!(
        "snake_cli profile={} {label} generations={generations} best_genome={} train_fitness={:.2} report_fitness={:.2} report_apples={} avg_steps={}",
        profile.name(),
        genome.key,
        genome.fitness.unwrap_or(0.0),
        report.fitness,
        report.eaten,
        report.steps
    );
    Ok(())
}
