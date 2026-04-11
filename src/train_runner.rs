use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use crate::{
    compat::{
        js::{run_neat_eval_worker, EvalBridgeError, EvalBridgeOptions},
        neat_python::{
            export_neat_python_genome_json, load_neat_python_config, new_rust_checkpoint_sink,
            new_rust_checkpointer, restore_rust_checkpoint, CheckpointError, Config, ConfigError,
        },
    },
    core::{
        evolution::{FitnessResult, Population, PopulationError},
        genomes::DefaultGenome,
        reporting::{mean, stdev},
    },
};

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRunnerOptions {
    pub neat_config_path: PathBuf,
    pub runtime_config_path: PathBuf,
    pub output_dir: PathBuf,
    pub seed: Option<u64>,
    pub feature_profile: Option<String>,
    pub eval_workers: Option<usize>,
    pub node_bin: String,
    pub resume_checkpoint: Option<PathBuf>,
}

impl TrainRunnerOptions {
    pub fn new(
        neat_config_path: impl Into<PathBuf>,
        runtime_config_path: impl Into<PathBuf>,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            neat_config_path: neat_config_path.into(),
            runtime_config_path: runtime_config_path.into(),
            output_dir: output_dir.into(),
            seed: None,
            feature_profile: None,
            eval_workers: None,
            node_bin: "node".to_string(),
            resume_checkpoint: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRuntimeConfig {
    pub generations: usize,
    pub eval_workers: usize,
    pub games_per_genome: usize,
    pub eval_timeout_sec: usize,
    pub max_eval_steps: usize,
    pub checkpoint_every: usize,
    pub eval_backend: TrainEvalBackend,
    pub eval_script: String,
    pub seed: u64,
    pub feature_profile: String,
    pub opponent_policy: String,
    pub opponent_policy_mix_json: Option<String>,
    pub opponent_genome: String,
    pub first_turn_policy: String,
    pub fixed_first_turn: String,
    pub continuous_series: bool,
    pub fitness_gold_scale: f64,
    pub fitness_gold_neutral_delta: f64,
    pub fitness_win_weight: f64,
    pub fitness_gold_weight: f64,
    pub fitness_win_neutral_rate: f64,
    pub early_stop_win_rate_cutoffs_json: Option<String>,
    pub early_stop_go_take_rate_cutoffs_json: Option<String>,
    pub native_inference_backend: Option<String>,
    pub winner_playoff_topk: usize,
    pub winner_playoff_games: usize,
    pub winner_playoff_eval_backend: TrainEvalBackend,
    pub winner_playoff_win_rate_tie_threshold: f64,
    pub winner_playoff_mean_gold_delta_tie_threshold: f64,
    pub winner_playoff_go_opp_min_count: usize,
    pub winner_playoff_go_take_rate_tie_threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainEvalBackend {
    JsWorker,
    CudaNative,
}

impl TrainEvalBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::JsWorker => "js_worker",
            Self::CudaNative => "cuda_native",
        }
    }

    fn parse(value: &str) -> Result<Self, TrainRunnerError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "js" | "js_worker" | "node" | "node_worker" | "neat_eval_worker" => {
                Ok(Self::JsWorker)
            }
            "cuda" | "cuda_native" | "rust_cuda" | "native_cuda" => Ok(Self::CudaNative),
            other => Err(TrainRunnerError::RuntimeConfig(format!(
                "unsupported eval backend '{other}'; expected one of: js_worker, cuda_native"
            ))),
        }
    }
}

impl TrainRuntimeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TrainRunnerError> {
        let mut fields = BTreeMap::new();
        load_runtime_fields(path.as_ref(), &mut fields, &mut Vec::new())?;
        Self::from_fields(&fields)
    }

    pub fn from_json_text(text: &str) -> Result<Self, TrainRunnerError> {
        let fields = parse_json_object_fields(text)?;
        Self::from_fields(&fields)
    }

    fn from_fields(fields: &BTreeMap<String, String>) -> Result<Self, TrainRunnerError> {
        let switch_seats = field_bool(fields, "switch_seats").unwrap_or(true);
        let eval_backend = TrainEvalBackend::parse(
            &field_string(fields, "eval_backend").unwrap_or_else(|| "js_worker".to_string()),
        )?;
        let opponent_policy_mix_json = fields
            .get("opponent_policy_mix")
            .filter(|value| !is_empty_json_array(value))
            .cloned();
        let early_stop_win_rate_cutoffs_json = fields
            .get("early_stop_win_rate_cutoffs")
            .filter(|value| !is_empty_json_array(value))
            .cloned();
        let early_stop_go_take_rate_cutoffs_json = fields
            .get("early_stop_go_take_rate_cutoffs")
            .filter(|value| !is_empty_json_array(value))
            .cloned();

        Ok(Self {
            generations: field_usize(fields, "generations").unwrap_or(1),
            eval_workers: field_usize(fields, "eval_workers").unwrap_or(1).max(1),
            games_per_genome: field_usize(fields, "games_per_genome").unwrap_or(3),
            eval_timeout_sec: field_usize(fields, "eval_timeout_sec").unwrap_or(0),
            max_eval_steps: field_usize(fields, "max_eval_steps").unwrap_or(600),
            checkpoint_every: field_usize(fields, "checkpoint_every").unwrap_or(0),
            eval_backend,
            eval_script: field_string(fields, "eval_script")
                .unwrap_or_else(|| "scripts/neat_eval_worker.mjs".to_string()),
            seed: field_usize(fields, "seed").unwrap_or(1) as u64,
            feature_profile: field_string(fields, "feature_profile")
                .unwrap_or_else(|| "material10".to_string())
                .trim()
                .to_ascii_lowercase(),
            opponent_policy: field_string(fields, "opponent_policy").unwrap_or_default(),
            opponent_policy_mix_json,
            opponent_genome: field_string(fields, "opponent_genome").unwrap_or_default(),
            first_turn_policy: if switch_seats {
                "alternate".to_string()
            } else {
                "fixed".to_string()
            },
            fixed_first_turn: "human".to_string(),
            continuous_series: field_bool(fields, "continuous_series").unwrap_or(true),
            fitness_gold_scale: field_f64(fields, "fitness_gold_scale").unwrap_or(1500.0),
            fitness_gold_neutral_delta: field_f64(fields, "fitness_gold_neutral_delta")
                .unwrap_or(0.0),
            fitness_win_weight: field_f64(fields, "fitness_win_weight").unwrap_or(0.85),
            fitness_gold_weight: field_f64(fields, "fitness_gold_weight").unwrap_or(0.15),
            fitness_win_neutral_rate: field_f64(fields, "fitness_win_neutral_rate").unwrap_or(0.50),
            early_stop_win_rate_cutoffs_json,
            early_stop_go_take_rate_cutoffs_json,
            native_inference_backend: field_string(fields, "native_inference_backend")
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty() && value != "off"),
            winner_playoff_topk: field_usize(fields, "winner_playoff_topk")
                .unwrap_or(5)
                .max(1),
            winner_playoff_games: field_usize(fields, "winner_playoff_games")
                .unwrap_or_else(|| field_usize(fields, "games_per_genome").unwrap_or(3))
                .max(1),
            winner_playoff_eval_backend: TrainEvalBackend::parse(
                &field_string(fields, "winner_playoff_eval_backend")
                    .unwrap_or_else(|| eval_backend.as_str().to_string()),
            )?,
            winner_playoff_win_rate_tie_threshold: field_f64(
                fields,
                "winner_playoff_win_rate_tie_threshold",
            )
            .unwrap_or(0.01),
            winner_playoff_mean_gold_delta_tie_threshold: field_f64(
                fields,
                "winner_playoff_mean_gold_delta_tie_threshold",
            )
            .unwrap_or(100.0),
            winner_playoff_go_opp_min_count: field_usize(fields, "winner_playoff_go_opp_min_count")
                .unwrap_or(100),
            winner_playoff_go_take_rate_tie_threshold: field_f64(
                fields,
                "winner_playoff_go_take_rate_tie_threshold",
            )
            .unwrap_or(0.02),
        })
    }

    pub fn validate(&self) -> Result<(), TrainRunnerError> {
        validate_eval_backend(self.eval_backend, "eval_backend")?;
        validate_eval_backend(
            self.winner_playoff_eval_backend,
            "winner_playoff_eval_backend",
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRunSummary {
    pub output_dir: PathBuf,
    pub winner_path: PathBuf,
    pub generations_requested: usize,
    pub best_genome_key: Option<i64>,
    pub best_fitness: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvalJob {
    generation: usize,
    genome_id: i64,
    genome_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct EvalJobResult {
    genome_id: i64,
    fitness: f64,
    win_rate: Option<f64>,
    mean_gold_delta: Option<f64>,
    go_take_rate: Option<f64>,
    go_fail_rate: Option<f64>,
    go_opportunity_count: Option<usize>,
    go_count: Option<usize>,
    go_games: Option<usize>,
    go_rate: Option<f64>,
    imitation_weighted_score: Option<f64>,
    eval_time_ms: Option<f64>,
    games: Option<usize>,
    eval_ok: Option<bool>,
    early_stop_triggered: Option<bool>,
    early_stop_reason: Option<String>,
    early_stop_cutoff_games: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct PlayoffSummary {
    winner_key: i64,
    path: PathBuf,
    results: Vec<EvalJobResult>,
}

#[derive(Debug)]
pub enum TrainRunnerError {
    Io(String),
    Config(ConfigError),
    RuntimeConfig(String),
    Checkpoint(CheckpointError),
    Population(PopulationError),
    Eval(EvalBridgeError),
    MissingFitness(i64),
    NoWinner,
}

impl fmt::Display for TrainRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "training I/O error: {message}"),
            Self::Config(err) => write!(f, "{err}"),
            Self::RuntimeConfig(message) => write!(f, "runtime config error: {message}"),
            Self::Checkpoint(err) => write!(f, "{err}"),
            Self::Population(err) => write!(f, "{err}"),
            Self::Eval(err) => write!(f, "{err}"),
            Self::MissingFitness(key) => write!(f, "fitness missing for genome {key}"),
            Self::NoWinner => write!(f, "training finished without a winner genome"),
        }
    }
}

impl Error for TrainRunnerError {}

impl From<ConfigError> for TrainRunnerError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<PopulationError> for TrainRunnerError {
    fn from(value: PopulationError) -> Self {
        Self::Population(value)
    }
}

impl From<EvalBridgeError> for TrainRunnerError {
    fn from(value: EvalBridgeError) -> Self {
        Self::Eval(value)
    }
}

impl From<CheckpointError> for TrainRunnerError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

pub fn run_kflower_training(
    options: &TrainRunnerOptions,
) -> Result<TrainRunSummary, TrainRunnerError> {
    let config = load_neat_python_config(&options.neat_config_path)?;
    let mut runtime = TrainRuntimeConfig::from_file(&options.runtime_config_path)?;
    if let Some(feature_profile) = &options.feature_profile {
        runtime.feature_profile = feature_profile.trim().to_ascii_lowercase();
    }
    if let Some(eval_workers) = options.eval_workers {
        runtime.eval_workers = eval_workers.max(1);
    }
    runtime.validate()?;
    let seed = options.seed.or(config.neat.seed).unwrap_or(runtime.seed);
    let project_root = project_root_from_runtime_config(&options.runtime_config_path);
    let eval_script = resolve_project_path(&project_root, &runtime.eval_script);
    let output_dir = absolute_path(&options.output_dir);
    let config_path = absolute_path(&options.neat_config_path);

    fs::create_dir_all(&output_dir).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    let metrics_path = output_dir.join("eval_metrics.ndjson");
    let generation_metrics_path = output_dir.join("generation_metrics.ndjson");
    let models_dir = output_dir.join("models");
    let temp_dir = output_dir.join(".tmp").join("genomes");
    fs::create_dir_all(&models_dir).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    fs::create_dir_all(&temp_dir).map_err(|err| TrainRunnerError::Io(err.to_string()))?;

    let mut population = if let Some(checkpoint_path) = &options.resume_checkpoint {
        restore_rust_checkpoint(checkpoint_path)?
    } else {
        Population::new(config, seed)?
    };
    let checkpoint_prefix = if runtime.checkpoint_every > 0 {
        Some(
            output_dir
                .join("checkpoints")
                .join("neat-rust-checkpoint-gen")
                .to_string_lossy()
                .to_string(),
        )
    } else {
        None
    };
    if let Some(prefix) = &checkpoint_prefix {
        population.checkpoint_sink = Some(new_rust_checkpoint_sink(
            Some(runtime.checkpoint_every),
            prefix.clone(),
            config_path.clone(),
        ));
    }
    let mut generation_index = if population.skip_first_evaluation {
        population.generation.saturating_add(1)
    } else {
        population.generation
    };
    let mut playoff_candidates = BTreeMap::new();

    let feature_profile = runtime.feature_profile.clone();
    let node_bin = command_path(&options.node_bin);
    let eval_script_for_run = eval_script.clone();
    let project_root_for_run = project_root.clone();
    let temp_dir_for_run = temp_dir.clone();
    let metrics_path_for_run = metrics_path.clone();
    let generation_metrics_path_for_run = generation_metrics_path.clone();

    let best = population.run(
        |genomes, config| -> FitnessResult {
            evaluate_generation(
                generation_index,
                genomes,
                config,
                &runtime,
                &feature_profile,
                &node_bin,
                &eval_script_for_run,
                &project_root_for_run,
                &temp_dir_for_run,
                &metrics_path_for_run,
                &generation_metrics_path_for_run,
            )
            .map_err(|err| err.to_string())?;
            update_playoff_candidates(
                &mut playoff_candidates,
                genomes,
                config,
                runtime.winner_playoff_topk,
            );
            generation_index += 1;
            Ok(())
        },
        Some(runtime.generations),
    )?;

    let fitness_winner = best
        .or_else(|| population.best_genome.clone())
        .ok_or(TrainRunnerError::NoWinner)?;
    playoff_candidates
        .entry(fitness_winner.key)
        .or_insert_with(|| fitness_winner.clone());
    let playoff = run_winner_playoff(
        &playoff_candidates,
        &population.config,
        &runtime,
        &feature_profile,
        &node_bin,
        &eval_script,
        &project_root,
        &output_dir,
    )?;
    let winner = playoff
        .as_ref()
        .and_then(|summary| playoff_candidates.get(&summary.winner_key).cloned())
        .unwrap_or(fitness_winner);
    let winner_path = models_dir.join("winner_genome.json");
    let winner_json = export_neat_python_genome_json(&winner, &population.config, &feature_profile);
    fs::write(&winner_path, winner_json).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    let winner_lineage_path = write_winner_lineage(&output_dir, &population, winner.key)?;
    let final_checkpoint_path = if let Some(prefix) = checkpoint_prefix {
        let evaluated_generation = generation_index.saturating_sub(1);
        let checkpointer =
            new_rust_checkpointer(Some(runtime.checkpoint_every), prefix, config_path.clone());
        let path = checkpointer.checkpoint_path(evaluated_generation);
        if checkpointer.should_save(evaluated_generation) && path.exists() {
            Some(path)
        } else {
            None
        }
    } else {
        None
    };
    write_run_summary(
        &output_dir,
        &winner_path,
        winner_lineage_path.as_deref(),
        final_checkpoint_path.as_deref(),
        &runtime,
        &feature_profile,
        playoff.as_ref(),
        runtime.generations,
        winner.key,
        winner.fitness,
    )?;

    Ok(TrainRunSummary {
        output_dir,
        winner_path,
        generations_requested: runtime.generations,
        best_genome_key: Some(winner.key),
        best_fitness: winner.fitness,
    })
}

fn evaluate_generation(
    generation: usize,
    genomes: &mut BTreeMap<i64, DefaultGenome>,
    config: &Config,
    runtime: &TrainRuntimeConfig,
    feature_profile: &str,
    node_bin: &str,
    eval_script: &Path,
    project_root: &Path,
    temp_dir: &Path,
    metrics_path: &Path,
    generation_metrics_path: &Path,
) -> Result<(), TrainRunnerError> {
    let generation_dir = temp_dir.join(format!("generation_{generation}"));
    fs::create_dir_all(&generation_dir).map_err(|err| TrainRunnerError::Io(err.to_string()))?;

    let mut jobs = Vec::new();
    for (genome_id, genome) in genomes.iter() {
        let genome_path = generation_dir.join(format!("genome_{genome_id}.json"));
        let json = export_neat_python_genome_json(genome, config, feature_profile);
        fs::write(&genome_path, json).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
        jobs.push(EvalJob {
            generation,
            genome_id: *genome_id,
            genome_path,
        });
    }

    let results = run_eval_jobs_parallel(
        jobs,
        runtime,
        runtime.eval_backend,
        node_bin,
        eval_script,
        project_root,
    )?;
    for result in &results {
        let Some(genome) = genomes.get_mut(&result.genome_id) else {
            return Err(TrainRunnerError::RuntimeConfig(format!(
                "eval result for unknown genome {}",
                result.genome_id
            )));
        };
        genome.fitness = Some(result.fitness);
        append_line(
            metrics_path,
            &format!(
                "{{\"generation\":{},\"genome_id\":{},\"fitness\":{},\"win_rate\":{},\"mean_gold_delta\":{},\"go_take_rate\":{},\"go_fail_rate\":{},\"go_opportunity_count\":{},\"go_count\":{},\"go_games\":{},\"go_rate\":{},\"imitation_weighted_score\":{},\"eval_time_ms\":{},\"games\":{},\"eval_ok\":{},\"early_stop_triggered\":{},\"early_stop_reason\":{},\"early_stop_cutoff_games\":{}}}",
                generation,
                result.genome_id,
                json_number(result.fitness),
                json_option_number(result.win_rate),
                json_option_number(result.mean_gold_delta),
                json_option_number(result.go_take_rate),
                json_option_number(result.go_fail_rate),
                json_option_usize(result.go_opportunity_count),
                json_option_usize(result.go_count),
                json_option_usize(result.go_games),
                json_option_number(result.go_rate),
                json_option_number(result.imitation_weighted_score),
                json_option_number(result.eval_time_ms),
                json_option_usize(result.games),
                result.eval_ok.unwrap_or(false),
                result.early_stop_triggered.unwrap_or(false),
                json_option_string(result.early_stop_reason.as_deref()),
                json_option_usize(result.early_stop_cutoff_games)
            ),
        )?;
    }

    let generation_metrics =
        generation_metrics_line(generation, genomes, &results, runtime.games_per_genome)?;
    append_line(generation_metrics_path, &generation_metrics)?;
    Ok(())
}

fn run_eval_jobs_parallel(
    jobs: Vec<EvalJob>,
    runtime: &TrainRuntimeConfig,
    backend: TrainEvalBackend,
    node_bin: &str,
    eval_script: &Path,
    project_root: &Path,
) -> Result<Vec<EvalJobResult>, TrainRunnerError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = runtime.eval_workers.max(1).min(job_count);
    let queue = Arc::new(Mutex::new(jobs.into_iter().collect::<VecDeque<_>>()));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let cancel = Arc::clone(&cancel);
        let tx = tx.clone();
        let runtime = runtime.clone();
        let node_bin = node_bin.to_string();
        let eval_script = eval_script.to_path_buf();
        let project_root = project_root.to_path_buf();

        handles.push(thread::spawn(move || loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let job = {
                let mut queue = queue.lock().expect("eval job queue lock poisoned");
                if cancel.load(Ordering::Relaxed) {
                    None
                } else {
                    queue.pop_front()
                }
            };
            let Some(job) = job else {
                break;
            };
            let result = run_eval_job(
                job,
                &runtime,
                backend,
                &node_bin,
                &eval_script,
                &project_root,
            );
            if result.is_err() {
                cancel.store(true, Ordering::Relaxed);
            }
            if tx.send(result).is_err() {
                cancel.store(true, Ordering::Relaxed);
                break;
            }
        }));
    }
    drop(tx);

    let mut results = Vec::new();
    let mut first_error = None;
    for result in rx {
        match result {
            Ok(result) => results.push(result),
            Err(err) => {
                cancel.store(true, Ordering::Relaxed);
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    let mut panicked = false;
    for handle in handles {
        if handle.join().is_err() {
            panicked = true;
        }
    }
    if panicked {
        return Err(TrainRunnerError::RuntimeConfig(
            "CPU eval worker thread panicked".to_string(),
        ));
    }
    if let Some(err) = first_error {
        return Err(err);
    }
    if results.len() != job_count {
        return Err(TrainRunnerError::RuntimeConfig(format!(
            "expected {job_count} eval results, got {}",
            results.len()
        )));
    }
    results.sort_by_key(|result| result.genome_id);
    Ok(results)
}

fn run_eval_job(
    job: EvalJob,
    runtime: &TrainRuntimeConfig,
    backend: TrainEvalBackend,
    node_bin: &str,
    eval_script: &Path,
    project_root: &Path,
) -> Result<EvalJobResult, TrainRunnerError> {
    if backend != TrainEvalBackend::JsWorker {
        return Err(unsupported_backend_error(backend));
    }

    let mut eval = EvalBridgeOptions::new(eval_script, &job.genome_path);
    eval.node_bin = node_bin.to_string();
    eval.working_dir = Some(project_root.to_path_buf());
    eval.games = runtime.games_per_genome;
    eval.timeout_millis = (runtime.eval_timeout_sec > 0)
        .then_some((runtime.eval_timeout_sec as u64).saturating_mul(1000));
    eval.seed = format!("gen{}_genome{}", job.generation, job.genome_id);
    eval.max_steps = runtime.max_eval_steps;
    eval.opponent_policy = non_empty_string(&runtime.opponent_policy);
    eval.opponent_policy_mix_json = runtime.opponent_policy_mix_json.clone();
    eval.opponent_genome_path = non_empty_string(&runtime.opponent_genome).map(PathBuf::from);
    eval.first_turn_policy = runtime.first_turn_policy.clone();
    eval.fixed_first_turn = runtime.fixed_first_turn.clone();
    eval.continuous_series = runtime.continuous_series;
    eval.fitness_gold_scale = runtime.fitness_gold_scale;
    eval.fitness_gold_neutral_delta = runtime.fitness_gold_neutral_delta;
    eval.fitness_win_weight = runtime.fitness_win_weight;
    eval.fitness_gold_weight = runtime.fitness_gold_weight;
    eval.fitness_win_neutral_rate = runtime.fitness_win_neutral_rate;
    eval.early_stop_win_rate_cutoffs_json = runtime.early_stop_win_rate_cutoffs_json.clone();
    eval.early_stop_go_take_rate_cutoffs_json =
        runtime.early_stop_go_take_rate_cutoffs_json.clone();
    eval.native_inference_backend = runtime.native_inference_backend.clone();

    let result = run_neat_eval_worker(&eval).map_err(TrainRunnerError::Eval)?;
    Ok(EvalJobResult {
        genome_id: job.genome_id,
        fitness: result.fitness.unwrap_or(0.0),
        win_rate: result.win_rate,
        mean_gold_delta: result.mean_gold_delta,
        go_take_rate: result.go_take_rate,
        go_fail_rate: result.go_fail_rate,
        go_opportunity_count: result.go_opportunity_count,
        go_count: result.go_count,
        go_games: result.go_games,
        go_rate: result.go_rate,
        imitation_weighted_score: result.imitation_weighted_score,
        eval_time_ms: result.eval_time_ms,
        games: result.games,
        eval_ok: result.eval_ok,
        early_stop_triggered: result.early_stop_triggered,
        early_stop_reason: result.early_stop_reason,
        early_stop_cutoff_games: result.early_stop_cutoff_games,
    })
}

fn update_playoff_candidates(
    candidates: &mut BTreeMap<i64, DefaultGenome>,
    genomes: &BTreeMap<i64, DefaultGenome>,
    config: &Config,
    topk: usize,
) {
    let limit = topk.max(1);
    for genome in genomes.values() {
        if genome.fitness.is_some() {
            candidates.insert(genome.key, genome.clone());
        }
    }

    let mut ranked: Vec<DefaultGenome> = candidates.values().cloned().collect();
    sort_genomes_by_fitness(&mut ranked, config);
    ranked.truncate(limit);
    candidates.clear();
    for genome in ranked {
        candidates.insert(genome.key, genome);
    }
}

fn run_winner_playoff(
    candidates: &BTreeMap<i64, DefaultGenome>,
    config: &Config,
    runtime: &TrainRuntimeConfig,
    feature_profile: &str,
    node_bin: &str,
    eval_script: &Path,
    project_root: &Path,
    output_dir: &Path,
) -> Result<Option<PlayoffSummary>, TrainRunnerError> {
    if candidates.is_empty() || runtime.winner_playoff_topk == 0 {
        return Ok(None);
    }

    let mut ranked: Vec<DefaultGenome> = candidates.values().cloned().collect();
    sort_genomes_by_fitness(&mut ranked, config);
    ranked.truncate(runtime.winner_playoff_topk.max(1));

    let playoff_dir = output_dir.join(".tmp").join("playoff");
    fs::create_dir_all(&playoff_dir).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    let mut jobs = Vec::new();
    for genome in &ranked {
        let genome_path = playoff_dir.join(format!("genome_{}.json", genome.key));
        let json = export_neat_python_genome_json(genome, config, feature_profile);
        fs::write(&genome_path, json).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
        jobs.push(EvalJob {
            generation: usize::MAX,
            genome_id: genome.key,
            genome_path,
        });
    }

    let mut playoff_runtime = runtime.clone();
    playoff_runtime.games_per_genome = runtime.winner_playoff_games;
    playoff_runtime.early_stop_win_rate_cutoffs_json = None;
    playoff_runtime.early_stop_go_take_rate_cutoffs_json = None;

    let results = run_eval_jobs_parallel(
        jobs,
        &playoff_runtime,
        runtime.winner_playoff_eval_backend,
        node_bin,
        eval_script,
        project_root,
    )?;
    let winner_key = select_playoff_winner(&results, runtime).ok_or(TrainRunnerError::NoWinner)?;
    let path = write_winner_playoff(output_dir, winner_key, &results, runtime)?;
    Ok(Some(PlayoffSummary {
        winner_key,
        path,
        results,
    }))
}

fn sort_genomes_by_fitness(genomes: &mut [DefaultGenome], config: &Config) {
    let ascending = config.neat.fitness_criterion.is_min();
    genomes.sort_by(|a, b| {
        let left = a.fitness.unwrap_or(if ascending {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        });
        let right = b.fitness.unwrap_or(if ascending {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        });
        if ascending {
            left.total_cmp(&right).then_with(|| a.key.cmp(&b.key))
        } else {
            right.total_cmp(&left).then_with(|| a.key.cmp(&b.key))
        }
    });
}

fn select_playoff_winner(results: &[EvalJobResult], runtime: &TrainRuntimeConfig) -> Option<i64> {
    let mut best = results.first()?;
    for result in &results[1..] {
        if playoff_result_is_better(result, best, runtime) {
            best = result;
        }
    }
    Some(best.genome_id)
}

fn playoff_result_is_better(
    candidate: &EvalJobResult,
    current: &EvalJobResult,
    runtime: &TrainRuntimeConfig,
) -> bool {
    let candidate_win = candidate.win_rate.unwrap_or(f64::NEG_INFINITY);
    let current_win = current.win_rate.unwrap_or(f64::NEG_INFINITY);
    if (candidate_win - current_win).abs() > runtime.winner_playoff_win_rate_tie_threshold {
        return candidate_win > current_win;
    }

    let candidate_gold = candidate.mean_gold_delta.unwrap_or(f64::NEG_INFINITY);
    let current_gold = current.mean_gold_delta.unwrap_or(f64::NEG_INFINITY);
    if (candidate_gold - current_gold).abs() > runtime.winner_playoff_mean_gold_delta_tie_threshold
    {
        return candidate_gold > current_gold;
    }

    let candidate_go = candidate.go_take_rate.unwrap_or(f64::NEG_INFINITY);
    let current_go = current.go_take_rate.unwrap_or(f64::NEG_INFINITY);
    if candidate
        .go_opportunity_count
        .unwrap_or(0)
        .min(current.go_opportunity_count.unwrap_or(0))
        >= runtime.winner_playoff_go_opp_min_count
    {
        if (candidate_go - current_go).abs() > runtime.winner_playoff_go_take_rate_tie_threshold {
            return candidate_go > current_go;
        }

        if candidate
            .go_count
            .unwrap_or(0)
            .min(current.go_count.unwrap_or(0))
            > 0
        {
            let candidate_go_fail = candidate.go_fail_rate.unwrap_or(f64::INFINITY);
            let current_go_fail = current.go_fail_rate.unwrap_or(f64::INFINITY);
            if (candidate_go_fail - current_go_fail).abs() > 1e-12 {
                return candidate_go_fail < current_go_fail;
            }
        }
    }

    if (candidate.fitness - current.fitness).abs() > f64::EPSILON {
        return candidate.fitness > current.fitness;
    }

    candidate.genome_id < current.genome_id
}

fn write_winner_playoff(
    output_dir: &Path,
    winner_key: i64,
    results: &[EvalJobResult],
    runtime: &TrainRuntimeConfig,
) -> Result<PathBuf, TrainRunnerError> {
    let path = output_dir.join("winner_playoff.json");
    let results_json = results
        .iter()
        .map(playoff_result_json)
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        "{{\"format_version\":\"neat_rust_winner_playoff_v1\",\"winner_genome_key\":{},\"topk\":{},\"games\":{},\"eval_backend\":\"{}\",\"win_rate_tie_threshold\":{},\"mean_gold_delta_tie_threshold\":{},\"go_opp_min_count\":{},\"go_take_rate_tie_threshold\":{},\"results\":[{}]}}\n",
        winner_key,
        runtime.winner_playoff_topk,
        runtime.winner_playoff_games,
        runtime.winner_playoff_eval_backend.as_str(),
        json_number(runtime.winner_playoff_win_rate_tie_threshold),
        json_number(runtime.winner_playoff_mean_gold_delta_tie_threshold),
        runtime.winner_playoff_go_opp_min_count,
        json_number(runtime.winner_playoff_go_take_rate_tie_threshold),
        results_json,
    );
    fs::write(&path, body).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    Ok(path)
}

fn playoff_result_json(result: &EvalJobResult) -> String {
    format!(
        "{{\"genome_id\":{},\"fitness\":{},\"win_rate\":{},\"mean_gold_delta\":{},\"go_take_rate\":{},\"go_fail_rate\":{},\"go_opportunity_count\":{},\"go_count\":{},\"go_games\":{},\"go_rate\":{},\"imitation_weighted_score\":{},\"eval_time_ms\":{},\"games\":{},\"eval_ok\":{}}}",
        result.genome_id,
        json_number(result.fitness),
        json_option_number(result.win_rate),
        json_option_number(result.mean_gold_delta),
        json_option_number(result.go_take_rate),
        json_option_number(result.go_fail_rate),
        json_option_usize(result.go_opportunity_count),
        json_option_usize(result.go_count),
        json_option_usize(result.go_games),
        json_option_number(result.go_rate),
        json_option_number(result.imitation_weighted_score),
        json_option_number(result.eval_time_ms),
        json_option_usize(result.games),
        result.eval_ok.unwrap_or(false),
    )
}

fn generation_metrics_line(
    generation: usize,
    genomes: &BTreeMap<i64, DefaultGenome>,
    results: &[EvalJobResult],
    requested_games: usize,
) -> Result<String, TrainRunnerError> {
    let mut best_key = None;
    let mut best_fitness = f64::NEG_INFINITY;
    let mut fitness_values = Vec::new();
    let mut node_counts = Vec::new();
    let mut enabled_connection_counts = Vec::new();
    let mut total_connection_counts = Vec::new();
    let mut best_nodes = 0usize;
    let mut best_enabled_connections = 0usize;
    let mut best_total_connections = 0usize;
    for (key, genome) in genomes {
        let fitness = genome
            .fitness
            .ok_or(TrainRunnerError::MissingFitness(*key))?;
        let node_count = genome.nodes.len();
        let enabled_connection_count = genome
            .connections
            .values()
            .filter(|connection| connection.enabled)
            .count();
        let total_connection_count = genome.connections.len();
        if fitness > best_fitness {
            best_fitness = fitness;
            best_key = Some(*key);
            best_nodes = node_count;
            best_enabled_connections = enabled_connection_count;
            best_total_connections = total_connection_count;
        }
        fitness_values.push(fitness);
        node_counts.push(node_count as f64);
        enabled_connection_counts.push(enabled_connection_count as f64);
        total_connection_counts.push(total_connection_count as f64);
    }

    let valid_record_count = results
        .iter()
        .filter(|result| result.eval_ok.unwrap_or(false))
        .count();
    let full_eval_record_count = results
        .iter()
        .filter(|result| {
            result.eval_ok.unwrap_or(false) && result.games.unwrap_or(0) >= requested_games
        })
        .count();
    let early_stop_record_count = results
        .iter()
        .filter(|result| {
            result.eval_ok.unwrap_or(false) && result.early_stop_triggered.unwrap_or(false)
        })
        .count();
    let invalid_record_count = results.len().saturating_sub(valid_record_count);
    let win_rates = option_values(results.iter().map(|result| result.win_rate));
    let mean_gold_deltas = option_values(results.iter().map(|result| result.mean_gold_delta));
    let go_take_rates = option_values(results.iter().map(|result| result.go_take_rate));
    let go_fail_rates = option_values(results.iter().map(|result| result.go_fail_rate));
    let imitation_scores =
        option_values(results.iter().map(|result| result.imitation_weighted_score));
    let eval_times = option_values(results.iter().map(|result| result.eval_time_ms));

    Ok(format!(
        "{{\"generation\":{},\"population_size\":{},\"valid_record_count\":{},\"full_eval_record_count\":{},\"early_stop_record_count\":{},\"invalid_record_count\":{},\"data_quality\":\"{}\",\"best_genome_id\":{},\"best_fitness\":{},\"mean_fitness\":{},\"std_fitness\":{},\"mean_win_rate\":{},\"mean_gold_delta\":{},\"mean_go_take_rate\":{},\"mean_go_fail_rate\":{},\"mean_imitation_weighted_score\":{},\"mean_eval_time_ms\":{},\"p90_eval_time_ms\":{},\"best_genome_nodes\":{},\"best_genome_connections\":{},\"best_genome_connections_total\":{},\"mean_nodes\":{},\"mean_connections\":{},\"mean_connections_total\":{}}}",
        generation,
        genomes.len(),
        valid_record_count,
        full_eval_record_count,
        early_stop_record_count,
        invalid_record_count,
        if valid_record_count > 0 {
            "valid_generation"
        } else {
            "invalid_generation"
        },
        best_key.unwrap_or(-1),
        json_number(best_fitness),
        json_number(mean(&fitness_values)),
        json_number(stdev(&fitness_values)),
        json_option_mean(&win_rates),
        json_option_mean(&mean_gold_deltas),
        json_option_mean(&go_take_rates),
        json_option_mean(&go_fail_rates),
        json_option_mean(&imitation_scores),
        json_option_mean(&eval_times),
        json_option_quantile(&eval_times, 0.9),
        best_nodes,
        best_enabled_connections,
        best_total_connections,
        json_number(mean(&node_counts)),
        json_number(mean(&enabled_connection_counts)),
        json_number(mean(&total_connection_counts)),
    ))
}

fn write_run_summary(
    output_dir: &Path,
    winner_path: &Path,
    winner_lineage_path: Option<&Path>,
    final_checkpoint_path: Option<&Path>,
    runtime: &TrainRuntimeConfig,
    feature_profile: &str,
    playoff: Option<&PlayoffSummary>,
    generations_requested: usize,
    winner_key: i64,
    best_fitness: Option<f64>,
) -> Result<(), TrainRunnerError> {
    let path = output_dir.join("run_summary.json");
    let selection_method = if playoff.is_some() {
        "winner_playoff"
    } else {
        "selection_fitness"
    };
    let line = format!(
        "{{\"format_version\":\"neat_rust_train_summary_v1\",\"generations_requested\":{},\"workers\":{},\"eval_backend\":\"{}\",\"winner_playoff_eval_backend\":\"{}\",\"feature_profile\":\"{}\",\"winner_genome_key\":{},\"winner_genome\":\"{}\",\"winner_lineage_path\":{},\"final_checkpoint_path\":{},\"best_fitness\":{},\"winner_playoff\":{},\"selection_method\":\"{}\"}}\n",
        generations_requested,
        runtime.eval_workers,
        runtime.eval_backend.as_str(),
        runtime.winner_playoff_eval_backend.as_str(),
        json_escape(feature_profile),
        winner_key,
        json_escape(&winner_path.to_string_lossy()),
        json_option_path(winner_lineage_path),
        json_option_path(final_checkpoint_path),
        json_option_number(best_fitness),
        playoff_summary_json(playoff),
        selection_method,
    );
    fs::write(path, line).map_err(|err| TrainRunnerError::Io(err.to_string()))
}

fn write_winner_lineage(
    output_dir: &Path,
    population: &Population,
    winner_key: i64,
) -> Result<Option<PathBuf>, TrainRunnerError> {
    let path = output_dir.join("winner_lineage.json");
    let mut entries = Vec::new();
    let mut stack = vec![(winner_key, 0usize)];
    let mut seen = Vec::new();
    while let Some((key, depth)) = stack.pop() {
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let parents = population
            .reproduction
            .ancestors
            .get(&key)
            .copied()
            .unwrap_or((None, None));
        entries.push(format!(
            "{{\"genome_key\":{},\"depth\":{},\"parent1\":{},\"parent2\":{}}}",
            key,
            depth,
            json_option_i64(parents.0),
            json_option_i64(parents.1)
        ));
        if let Some(parent) = parents.0 {
            stack.push((parent, depth + 1));
        }
        if let Some(parent) = parents.1 {
            stack.push((parent, depth + 1));
        }
    }

    let body = format!(
        "{{\"format_version\":\"neat_rust_winner_lineage_v1\",\"winner_genome_key\":{},\"ancestors\":[{}]}}\n",
        winner_key,
        entries.join(",")
    );
    fs::write(&path, body).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    Ok(Some(path))
}

fn append_line(path: &Path, line: &str) -> Result<(), TrainRunnerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    file.write_all(line.as_bytes())
        .map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    file.write_all(b"\n")
        .map_err(|err| TrainRunnerError::Io(err.to_string()))
}

fn load_runtime_fields(
    path: &Path,
    fields: &mut BTreeMap<String, String>,
    seen: &mut Vec<PathBuf>,
) -> Result<(), TrainRunnerError> {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    if seen.iter().any(|item| item == &absolute) {
        return Ok(());
    }
    seen.push(absolute.clone());

    let text =
        fs::read_to_string(&absolute).map_err(|err| TrainRunnerError::Io(err.to_string()))?;
    let mut local = parse_json_object_fields(&text)?;
    if let Some(extends) = local.get("extends").cloned() {
        for child in parse_extends_list(&extends) {
            let child_path = resolve_runtime_path(&absolute, &child);
            load_runtime_fields(&child_path, fields, seen)?;
        }
    }
    local.remove("extends");
    fields.extend(local);
    Ok(())
}

fn resolve_runtime_path(current_file: &Path, child: &str) -> PathBuf {
    let child_path = PathBuf::from(child);
    if child_path.is_absolute() {
        child_path
    } else {
        current_file
            .parent()
            .map(|parent| parent.join(child_path))
            .unwrap_or_else(|| PathBuf::from(child))
    }
}

fn parse_extends_list(raw: &str) -> Vec<String> {
    if let Some(value) = parse_json_string_literal(raw) {
        return vec![value];
    }
    if !raw.trim_start().starts_with('[') {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx = 0usize;
    while let Some(start) = raw[idx..].find('"') {
        let start = idx + start;
        let Some((value, end)) = parse_json_string_at(raw, start) else {
            break;
        };
        out.push(value);
        idx = end;
    }
    out
}

fn parse_json_object_fields(text: &str) -> Result<BTreeMap<String, String>, TrainRunnerError> {
    let mut fields = BTreeMap::new();
    let mut idx = text
        .find('{')
        .ok_or_else(|| TrainRunnerError::RuntimeConfig("expected JSON object".to_string()))?
        + 1;

    loop {
        idx = skip_json_ws(text, idx);
        if idx >= text.len() {
            break;
        }
        if text[idx..].starts_with('}') {
            break;
        }
        let Some((key, after_key)) = parse_json_string_at(text, idx) else {
            return Err(TrainRunnerError::RuntimeConfig(
                "expected JSON object key".to_string(),
            ));
        };
        idx = skip_json_ws(text, after_key);
        if !text[idx..].starts_with(':') {
            return Err(TrainRunnerError::RuntimeConfig(format!(
                "expected ':' after runtime key {key}"
            )));
        }
        idx = skip_json_ws(text, idx + 1);
        let (raw_value, after_value) = read_json_value(text, idx)?;
        fields.insert(key, raw_value.trim().to_string());
        idx = skip_json_ws(text, after_value);
        if idx < text.len() && text[idx..].starts_with(',') {
            idx += 1;
        }
    }

    Ok(fields)
}

fn read_json_value(text: &str, start: usize) -> Result<(String, usize), TrainRunnerError> {
    let mut idx = start;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    while idx < text.len() {
        let ch = text[idx..].chars().next().unwrap_or('\0');
        let next = idx + ch.len_utf8();
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            idx = next;
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' | '{' => depth += 1,
            ']' | '}' if depth > 0 => depth -= 1,
            ',' | '}' if depth == 0 => return Ok((text[start..idx].to_string(), idx)),
            _ => {}
        }
        idx = next;
    }

    if start < idx {
        Ok((text[start..idx].to_string(), idx))
    } else {
        Err(TrainRunnerError::RuntimeConfig(
            "expected JSON value".to_string(),
        ))
    }
}

fn skip_json_ws(text: &str, mut idx: usize) -> usize {
    while idx < text.len() {
        let ch = text[idx..].chars().next().unwrap_or('\0');
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn parse_json_string_at(text: &str, start: usize) -> Option<(String, usize)> {
    if !text[start..].starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    let mut idx = start + 1;
    while idx < text.len() {
        let ch = text[idx..].chars().next()?;
        idx += ch.len_utf8();
        if escaped {
            match ch {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some((out, idx));
        } else {
            out.push(ch);
        }
    }
    None
}

fn parse_json_string_literal(raw: &str) -> Option<String> {
    let start = raw.find('"')?;
    parse_json_string_at(raw, start).map(|(value, _)| value)
}

fn field_string(fields: &BTreeMap<String, String>, key: &str) -> Option<String> {
    parse_json_string_literal(fields.get(key)?)
}

fn validate_eval_backend(
    backend: TrainEvalBackend,
    field_name: &str,
) -> Result<(), TrainRunnerError> {
    if backend == TrainEvalBackend::JsWorker {
        return Ok(());
    }
    Err(TrainRunnerError::RuntimeConfig(format!(
        "{field_name}={} is not available for k-flower train/playoff yet; the current path runs full duel simulation through scripts/neat_eval_worker.mjs in Node. Native CUDA is implemented for Rust GPUCTRNNEvaluator/GPUIZNNEvaluator, but train/playoff need a Rust-native k-flower evaluator or a persistent JS<->Rust inference bridge before CUDA can accelerate them.",
        backend.as_str()
    )))
}

fn unsupported_backend_error(backend: TrainEvalBackend) -> TrainRunnerError {
    TrainRunnerError::RuntimeConfig(format!(
        "train/playoff eval backend '{}' is not implemented for the current k-flower runner",
        backend.as_str()
    ))
}

fn field_bool(fields: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    match fields.get(key)?.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn field_usize(fields: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    let value = field_f64(fields, key)?;
    if value.is_finite() && value >= 0.0 {
        Some(value as usize)
    } else {
        None
    }
}

fn field_f64(fields: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    fields.get(key)?.trim().parse::<f64>().ok()
}

fn project_root_from_runtime_config(path: &Path) -> PathBuf {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    absolute
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn resolve_project_path(project_root: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        project_root.join(path)
    }
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn command_path(value: &str) -> String {
    if value.contains('\\') || value.contains('/') {
        absolute_path(Path::new(value))
            .to_string_lossy()
            .to_string()
    } else {
        value.to_string()
    }
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn is_empty_json_array(value: &str) -> bool {
    let compact = value
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    compact == "[]" || compact == "null"
}

fn json_option_number(value: Option<f64>) -> String {
    value.map(json_number).unwrap_or_else(|| "null".to_string())
}

fn option_values(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().filter(|value| value.is_finite()).collect()
}

fn json_option_mean(values: &[f64]) -> String {
    if values.is_empty() {
        "null".to_string()
    } else {
        json_number(mean(values))
    }
}

fn json_option_quantile(values: &[f64], q: f64) -> String {
    if values.is_empty() {
        return "null".to_string();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len().saturating_sub(1)) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    json_number(sorted[index.min(sorted.len() - 1)])
}

fn json_option_i64(value: Option<i64>) -> String {
    value
        .map(|number| number.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_option_path(value: Option<&Path>) -> String {
    value
        .map(|path| format!("\"{}\"", json_escape(&path.to_string_lossy())))
        .unwrap_or_else(|| "null".to_string())
}

fn json_option_string(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn json_option_usize(value: Option<usize>) -> String {
    value
        .map(|number| number.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn playoff_summary_json(value: Option<&PlayoffSummary>) -> String {
    let Some(summary) = value else {
        return "null".to_string();
    };
    format!(
        "{{\"winner_genome_key\":{},\"path\":\"{}\",\"result_count\":{}}}",
        summary.winner_key,
        json_escape(&summary.path.to_string_lossy()),
        summary.results.len()
    )
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}
