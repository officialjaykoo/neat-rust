use std::env;
use std::fs;
use std::path::Path;
use std::process;

use neat_rust::{
    algorithm::{DefaultGenome, XorShiftRng},
    io::{export_neat_genome_json, load_neat_config, Config},
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

    let config_path = arg_value(&args, "--config")
        .or_else(|| arg_value(&args, "--config-feedforward"))
        .ok_or_else(|| "missing --config <path>".to_string())?;

    let config = load_neat_config(&config_path).map_err(|err| err.to_string())?;
    print_config_summary(&config_path, &config);

    let out_path = arg_value(&args, "--out");
    if let Some(out_path) = out_path.as_deref() {
        let seed = arg_value(&args, "--seed")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1);
        let feature_profile = arg_value(&args, "--feature-profile").unwrap_or_default();
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut genome = DefaultGenome::new(0);
        genome
            .configure_new(&config.genome, &mut rng)
            .map_err(|err| err.to_string())?;
        let json = export_neat_genome_json(&genome, &config, feature_profile);
        if let Some(parent) = Path::new(&out_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
        }
        fs::write(out_path, json).map_err(|err| err.to_string())?;
        println!("genome_nodes={}", genome.nodes.len());
        println!("genome_connections={}", genome.connections.len());
        println!("out={out_path}");
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
    println!("neat-rust-inspect");
    println!();
    println!("Usage:");
    println!("  neat-rust-inspect --config <path>");
    println!("  neat-rust-inspect --config-feedforward <path>");
    println!(
        "  neat-rust-inspect --config <path> --out <path> [--feature-profile <name>] [--seed <n>]"
    );
    println!();
    println!("Current milestone:");
    println!("  Parse a NEAT TOML config and export a Rust NEAT genome JSON.");
}

fn print_config_summary(path: &str, config: &Config) {
    println!("config={path}");
    println!("fitness_criterion={}", config.neat.fitness_criterion);
    println!("fitness_threshold={}", config.neat.fitness_threshold);
    println!("pop_size={}", config.neat.pop_size);
    println!("reset_on_extinction={}", config.neat.reset_on_extinction);
    println!(
        "no_fitness_termination={}",
        config.neat.no_fitness_termination
    );
    println!("num_inputs={}", config.genome.num_inputs);
    println!("num_outputs={}", config.genome.num_outputs);
    println!("num_hidden={}", config.genome.num_hidden);
    println!("feed_forward={}", config.genome.feed_forward);
    println!("initial_connection={}", config.genome.initial_connection);
    println!("input_keys={:?}", config.input_keys());
    println!("output_keys={:?}", config.output_keys());
    println!(
        "activation_default={}",
        config.genome.activation.default_label()
    );
    println!(
        "activation_options={}",
        config.genome.activation.options_label()
    );
    println!(
        "aggregation_default={}",
        config.genome.aggregation.default_label()
    );
    println!(
        "compatibility_threshold={}",
        config.species_set.compatibility_threshold
    );
    println!("elitism={}", config.reproduction.elitism);
    println!(
        "survival_threshold={}",
        config.reproduction.survival_threshold
    );
    println!(
        "memory_gate_enabled_default={}",
        config.genome.memory_gate_enabled.default
    );
}
