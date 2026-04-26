use std::env;
use std::error::Error;

use neat_rust::{
    algorithm::{
        BootstrapStrategy, DefaultGenome, FitnessError, FitnessEvaluator, GenomeId, Population,
    },
    io::Config,
    network::RecurrentNetwork,
};

const MAX_STEPS: usize = 360;
const DT: f64 = 0.05;
const GRAVITY: f64 = -0.018;
const MAIN_THRUST: f64 = 0.038;
const SIDE_THRUST: f64 = 0.012;
const TORQUE: f64 = 0.030;
const DAMPING: f64 = 0.995;

#[derive(Debug, Clone, Copy)]
struct LanderState {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    angle: f64,
    angular_velocity: f64,
}

#[derive(Debug, Clone, Copy)]
enum LateralAction {
    Left,
    Right,
    None,
}

struct LunarLanderEvaluator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LanderConfigProfile {
    Plain,
    NodeGru,
}

impl LanderConfigProfile {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("plain").to_ascii_lowercase().as_str() {
            "plain" | "base" | "no-gru" | "no_gru" => Ok(Self::Plain),
            "gru" | "node-gru" | "node_gru" | "nodegru" => Ok(Self::NodeGru),
            other => Err(format!(
                "unknown lunar lander config profile {other:?}; use plain or node-gru"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::NodeGru => "node-gru",
        }
    }

    fn config_text(self) -> &'static str {
        match self {
            Self::Plain => include_str!("lunar_lander_config.toml"),
            Self::NodeGru => include_str!("lunar_lander_node_gru_config.toml"),
        }
    }
}

impl FitnessEvaluator for LunarLanderEvaluator {
    fn evaluate_genome(
        &mut self,
        _genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let starts = [
            LanderState::new(-0.45, 1.20, 0.018, -0.010, 0.10, 0.00),
            LanderState::new(0.42, 1.15, -0.014, -0.005, -0.08, 0.01),
            LanderState::new(-0.15, 1.05, 0.005, 0.000, 0.20, -0.01),
            LanderState::new(0.20, 1.30, -0.010, -0.012, -0.18, 0.00),
        ];
        let mut total = 0.0;
        for start in starts {
            let mut network = RecurrentNetwork::create(genome, &config.genome)
                .map_err(|err| FitnessError::new(err.to_string()))?;
            total += run_episode(start, &mut network)?;
        }
        Ok(total / starts.len() as f64)
    }
}

impl LanderState {
    fn new(x: f64, y: f64, vx: f64, vy: f64, angle: f64, angular_velocity: f64) -> Self {
        Self {
            x,
            y,
            vx,
            vy,
            angle,
            angular_velocity,
        }
    }

    fn inputs(self) -> [f64; 6] {
        [
            self.x,
            (self.y - 0.5).clamp(-1.0, 1.0),
            self.vx * 8.0,
            self.vy * 8.0,
            self.angle,
            self.angular_velocity * 8.0,
        ]
    }

    fn landed(self) -> bool {
        self.y <= 0.0
            && self.x.abs() < 0.20
            && self.vx.abs() < 0.045
            && self.vy.abs() < 0.055
            && self.angle.abs() < 0.20
            && self.angular_velocity.abs() < 0.050
    }

    fn crashed(self) -> bool {
        self.y <= 0.0 || self.x.abs() > 1.6 || self.y > 1.8 || self.angle.abs() > 1.2
    }
}

fn run_episode(
    mut state: LanderState,
    network: &mut RecurrentNetwork,
) -> Result<f64, FitnessError> {
    network.reset();
    let mut shaping = 0.0;
    for step in 0..MAX_STEPS {
        let outputs = network
            .activate(&state.inputs())
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let main = outputs.first().copied().unwrap_or(0.0) > 0.0;
        let lateral = match outputs.get(1).copied().unwrap_or(0.0) {
            value if value < -0.35 => LateralAction::Left,
            value if value > 0.35 => LateralAction::Right,
            _ => LateralAction::None,
        };
        state = step_lander(state, main, lateral);
        shaping += step_reward(state, main, lateral);

        if state.crashed() {
            let terminal = if state.landed() { 350.0 } else { -120.0 };
            return Ok((450.0 + shaping + terminal + step as f64 * 0.2).max(0.0));
        }
    }
    Ok((450.0 + shaping - 80.0).max(0.0))
}

fn step_lander(mut state: LanderState, main: bool, lateral: LateralAction) -> LanderState {
    let mut ax = 0.0;
    let mut ay = GRAVITY;
    let mut angular_acc = 0.0;

    if main {
        ax += state.angle.sin() * MAIN_THRUST;
        ay += state.angle.cos() * MAIN_THRUST;
    }
    match lateral {
        LateralAction::Left => {
            ax -= SIDE_THRUST;
            angular_acc += TORQUE;
        }
        LateralAction::Right => {
            ax += SIDE_THRUST;
            angular_acc -= TORQUE;
        }
        LateralAction::None => {}
    }

    state.vx = (state.vx + ax) * DAMPING;
    state.vy = (state.vy + ay) * DAMPING;
    state.angular_velocity = (state.angular_velocity + angular_acc) * DAMPING;
    state.x += state.vx * DT;
    state.y += state.vy * DT;
    state.angle += state.angular_velocity * DT;
    state
}

fn step_reward(state: LanderState, main: bool, lateral: LateralAction) -> f64 {
    let control_cost = if main { 0.10 } else { 0.0 }
        + if matches!(lateral, LateralAction::None) {
            0.0
        } else {
            0.04
        };
    1.0 - (state.x.abs() * 2.0)
        - (state.vx.abs() * 6.0)
        - (state.vy.abs() * 6.0)
        - (state.angle.abs() * 1.5)
        - (state.angular_velocity.abs() * 4.0)
        - control_cost
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
    let profile = LanderConfigProfile::parse(args.get(2).map(String::as_str))?;
    let config = Config::from_toml_str(profile.config_text())?;
    let mut evaluator = LunarLanderEvaluator;
    let mut best = run_population(
        Population::new(config.clone(), 84)?,
        &mut evaluator,
        generations,
    )?;

    println!(
        "lunar_lander profile={} round=0 bootstrap=false generations={} best_genome={} fitness={:.2}",
        profile.name(),
        generations,
        best.key,
        best.fitness.unwrap_or(0.0)
    );

    for round in 1..=bootstrap_rounds {
        let strategy = BootstrapStrategy::from_champion(best.clone(), 0.5);
        let candidate = run_population(
            Population::new_with_bootstrap(config.clone(), 84 + round as u64, strategy)?,
            &mut evaluator,
            generations,
        )?;
        println!(
            "lunar_lander profile={} round={} bootstrap=true generations={} best_genome={} fitness={:.2}",
            profile.name(),
            round,
            generations,
            candidate.key,
            candidate.fitness.unwrap_or(0.0)
        );
        if candidate.fitness.unwrap_or(f64::NEG_INFINITY)
            > best.fitness.unwrap_or(f64::NEG_INFINITY)
        {
            best = candidate;
        }
    }

    println!(
        "lunar_lander profile={} final generations_per_round={} bootstrap_rounds={} best_genome={} fitness={:.2}",
        profile.name(),
        generations,
        bootstrap_rounds,
        best.key,
        best.fitness.unwrap_or(0.0)
    );
    Ok(())
}

fn run_population(
    mut population: Population,
    evaluator: &mut LunarLanderEvaluator,
    generations: usize,
) -> Result<DefaultGenome, Box<dyn Error>> {
    Ok(population
        .run_with_evaluator(evaluator, Some(generations))?
        .expect("population should keep a champion"))
}
