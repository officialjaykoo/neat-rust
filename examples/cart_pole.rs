use std::env;
use std::error::Error;

use neat_rust::{
    algorithm::{DefaultGenome, FitnessError, FitnessEvaluator, GenomeId, Population},
    io::Config,
    network::FeedForwardNetwork,
};

const GRAVITY: f64 = 9.8;
const MASS_CART: f64 = 1.0;
const MASS_POLE: f64 = 0.1;
const TOTAL_MASS: f64 = MASS_CART + MASS_POLE;
const LENGTH: f64 = 0.5;
const POLE_MASS_LENGTH: f64 = MASS_POLE * LENGTH;
const FORCE_MAG: f64 = 10.0;
const TAU: f64 = 0.02;
const X_THRESHOLD: f64 = 2.4;
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
const MAX_STEPS: usize = 500;

#[derive(Debug, Clone, Copy)]
struct CartPoleState {
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
}

struct CartPoleEvaluator;

impl FitnessEvaluator for CartPoleEvaluator {
    fn evaluate_genome(
        &mut self,
        _genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let starts = [
            CartPoleState::new(0.00, 0.00, 0.020, 0.00),
            CartPoleState::new(0.02, 0.00, -0.015, 0.01),
            CartPoleState::new(-0.02, 0.01, 0.010, -0.01),
            CartPoleState::new(0.00, -0.01, -0.020, 0.00),
        ];
        let mut total = 0.0;
        for start in starts {
            let mut network = FeedForwardNetwork::create(genome, &config.genome)
                .map_err(|err| FitnessError::new(err.to_string()))?;
            total += run_episode(start, &mut network)?;
        }
        Ok(total / starts.len() as f64)
    }
}

impl CartPoleState {
    fn new(x: f64, x_dot: f64, theta: f64, theta_dot: f64) -> Self {
        Self {
            x,
            x_dot,
            theta,
            theta_dot,
        }
    }

    fn inputs(self) -> [f64; 4] {
        [
            self.x / X_THRESHOLD,
            self.x_dot / 3.0,
            self.theta / THETA_THRESHOLD,
            self.theta_dot / 3.0,
        ]
    }

    fn failed(self) -> bool {
        self.x.abs() > X_THRESHOLD || self.theta.abs() > THETA_THRESHOLD
    }
}

fn run_episode(
    mut state: CartPoleState,
    network: &mut FeedForwardNetwork,
) -> Result<f64, FitnessError> {
    let mut fitness = 0.0;
    for step in 0..MAX_STEPS {
        let output = network
            .activate(&state.inputs())
            .map_err(|err| FitnessError::new(err.to_string()))?
            .first()
            .copied()
            .unwrap_or(0.0);
        let force = if output >= 0.0 { FORCE_MAG } else { -FORCE_MAG };
        state = step_cart_pole(state, force);
        if state.failed() {
            fitness += step as f64 / MAX_STEPS as f64;
            break;
        }
        fitness += 1.0;
    }
    Ok(fitness)
}

fn step_cart_pole(state: CartPoleState, force: f64) -> CartPoleState {
    let costheta = state.theta.cos();
    let sintheta = state.theta.sin();
    let temp = (force + POLE_MASS_LENGTH * state.theta_dot.powi(2) * sintheta) / TOTAL_MASS;
    let theta_acc = (GRAVITY * sintheta - costheta * temp)
        / (LENGTH * (4.0 / 3.0 - MASS_POLE * costheta.powi(2) / TOTAL_MASS));
    let x_acc = temp - POLE_MASS_LENGTH * theta_acc * costheta / TOTAL_MASS;

    CartPoleState {
        x: state.x + TAU * state.x_dot,
        x_dot: state.x_dot + TAU * x_acc,
        theta: state.theta + TAU * state.theta_dot,
        theta_dot: state.theta_dot + TAU * theta_acc,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let generations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12);
    let config = Config::from_toml_str(include_str!("cart_pole_config.toml"))?;
    let mut population = Population::new(config, 42)?;
    let mut evaluator = CartPoleEvaluator;
    let best = population
        .run_with_evaluator(&mut evaluator, Some(generations))?
        .expect("population should keep a champion");

    println!(
        "cart_pole generations={} best_genome={} fitness={:.2}",
        generations,
        best.key,
        best.fitness.unwrap_or(0.0)
    );
    Ok(())
}
