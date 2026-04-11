use std::env;
use std::path::PathBuf;
use std::process;

use neat_rust::{
    compat::js::default_node_bin,
    runtime::kflower::{run_kflower_training, TrainRunnerOptions},
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(());
    }

    let neat_config_path = arg_value(&args, "--config-feedforward")
        .or_else(|| arg_value(&args, "--config"))
        .ok_or_else(|| "missing --config-feedforward <path>".to_string())?;
    let runtime_config_path = arg_value(&args, "--runtime-config")
        .ok_or_else(|| "missing --runtime-config <path>".to_string())?;
    let output_dir = arg_value(&args, "--output-dir")
        .ok_or_else(|| "missing --output-dir <path>".to_string())?;

    let mut options = TrainRunnerOptions::new(
        PathBuf::from(neat_config_path),
        PathBuf::from(runtime_config_path),
        PathBuf::from(output_dir),
    );
    options.seed = arg_value(&args, "--seed").and_then(|value| value.parse::<u64>().ok());
    options.feature_profile = arg_value(&args, "--feature-profile");
    options.eval_workers = arg_value(&args, "--workers")
        .or_else(|| arg_value(&args, "--eval-workers"))
        .and_then(|value| value.parse::<usize>().ok());
    options.node_bin = arg_value(&args, "--node-bin").unwrap_or_else(default_node_bin);
    options.resume_checkpoint = arg_value(&args, "--resume-checkpoint").map(PathBuf::from);

    let summary = run_kflower_training(&options).map_err(|err| err.to_string())?;
    println!("output_dir={}", summary.output_dir.display());
    println!("winner={}", summary.winner_path.display());
    if let Some(key) = summary.best_genome_key {
        println!("best_genome_key={key}");
    }
    if let Some(fitness) = summary.best_fitness {
        println!("best_fitness={fitness}");
    }
    Ok(())
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    let prefixed = format!("{key}=");
    for (idx, arg) in args.iter().enumerate() {
        if arg == key {
            return args.get(idx + 1).cloned();
        }
        if let Some(value) = arg.strip_prefix(&prefixed) {
            return Some(value.to_string());
        }
    }
    None
}

fn print_help() {
    println!("neat-train-rs");
    println!();
    println!("Usage:");
    println!("  neat-train-rs --config-feedforward <ini> --runtime-config <json> --output-dir <dir> [--seed <n>] [--feature-profile <name>] [--workers <n>] [--resume-checkpoint <path>]");
    println!();
    println!("Current milestone:");
    println!("  Run the Rust NEAT population loop while evaluating genomes through scripts/neat_eval_worker.mjs.");
}
