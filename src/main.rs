use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Duration;

use neat_rust::{
    compat::{
        js::{
            default_node_bin, run_neat_eval_worker, BridgeEarlyStopConfig, BridgeGameCount,
            BridgeNativeInferenceBackend, BridgeOpponent, BridgeStepCount, BridgeTurnPolicy,
            EvalBridgeOptions, EvalSeed, NodeCommand,
        },
        neat_format::{export_neat_genome_json, load_neat_config, Config},
    },
    core::{attributes::XorShiftRng, genomes::DefaultGenome},
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
        let feature_profile =
            arg_value(&args, "--feature-profile").unwrap_or_else(|| "".to_string());
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
        fs::write(&out_path, json).map_err(|err| err.to_string())?;
        println!("genome_nodes={}", genome.nodes.len());
        println!("genome_connections={}", genome.connections.len());
        println!("out={out_path}");
    }

    if let Some(eval_worker) = arg_value(&args, "--eval-worker") {
        let out_path = out_path
            .as_deref()
            .ok_or_else(|| "--eval-worker requires --out <genome-json-path>".to_string())?;
        let mut eval = EvalBridgeOptions::new(eval_worker, out_path);
        eval.node_bin =
            NodeCommand::new(arg_value(&args, "--node-bin").unwrap_or_else(default_node_bin));
        eval.games = BridgeGameCount::new(
            arg_value(&args, "--games")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(eval.games.get()),
        );
        eval.timeout = arg_value(&args, "--eval-timeout-sec")
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .map(Duration::from_secs);
        eval.seed = EvalSeed::new(
            arg_value(&args, "--eval-seed")
                .or_else(|| arg_value(&args, "--seed"))
                .unwrap_or_else(|| eval.seed.as_str().to_string()),
        );
        eval.max_steps = BridgeStepCount::new(
            arg_value(&args, "--max-steps")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(eval.max_steps.get()),
        );
        eval.opponent = BridgeOpponent::from_parts(
            arg_value(&args, "--opponent-policy"),
            arg_value(&args, "--opponent-policy-mix"),
        )
        .map_err(|err| err.to_string())?;
        eval.opponent_genome_path = arg_value(&args, "--opponent-genome").map(Into::into);
        eval.turn_policy = BridgeTurnPolicy::from_parts(
            arg_value(&args, "--first-turn-policy"),
            arg_value(&args, "--fixed-first-turn"),
        )
        .map_err(|err| err.to_string())?;
        eval.continuous_series = arg_value(&args, "--continuous-series")
            .map(|value| parse_bool_like(&value))
            .unwrap_or(eval.continuous_series);
        eval.fitness_gold_scale = arg_value(&args, "--fitness-gold-scale")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(eval.fitness_gold_scale);
        eval.fitness_gold_neutral_delta = arg_value(&args, "--fitness-gold-neutral-delta")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(eval.fitness_gold_neutral_delta);
        eval.fitness_win_weight = arg_value(&args, "--fitness-win-weight")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(eval.fitness_win_weight);
        eval.fitness_gold_weight = arg_value(&args, "--fitness-gold-weight")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(eval.fitness_gold_weight);
        eval.fitness_win_neutral_rate = arg_value(&args, "--fitness-win-neutral-rate")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(eval.fitness_win_neutral_rate);
        eval.early_stop = BridgeEarlyStopConfig::from_parts(
            arg_value(&args, "--early-stop-win-rate-cutoffs"),
            arg_value(&args, "--early-stop-go-take-rate-cutoffs"),
        )
        .map_err(|err| err.to_string())?;
        eval.native_inference_backend = BridgeNativeInferenceBackend::from_optional(arg_value(
            &args,
            "--native-inference-backend",
        ));

        let result = run_neat_eval_worker(&eval).map_err(|err| err.to_string())?;
        println!("eval_status={:?}", result.status_code);
        if let Some(eval_ok) = result.eval_ok {
            println!("eval_ok={eval_ok}");
        }
        if let Some(fitness) = result.fitness {
            println!("fitness={fitness}");
        }
        if let Some(win_rate) = result.win_rate {
            println!("win_rate={win_rate}");
        }
        if let Some(games) = result.games {
            println!("eval_games={games}");
        }
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
    println!("  neat-rust-inspect --config <path> --out <path> --eval-worker scripts/kflower_eval_worker.mjs --opponent-policy <policy> [--eval-timeout-sec <n>]");
    println!();
    println!("Current milestone:");
    println!("  Parse a NEAT INI config, export a JS-compatible genome JSON, and optionally call the JS eval worker.");
}

fn parse_bool_like(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
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
    println!("activation_default={}", config.genome.activation.default);
    println!(
        "activation_options={}",
        config.genome.activation.options.join(" ")
    );
    println!("aggregation_default={}", config.genome.aggregation.default);
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
