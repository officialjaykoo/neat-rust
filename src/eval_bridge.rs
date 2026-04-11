use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct EvalBridgeOptions {
    pub node_bin: String,
    pub worker_script: PathBuf,
    pub working_dir: Option<PathBuf>,
    pub genome_path: PathBuf,
    pub timeout_millis: Option<u64>,
    pub opponent_genome_path: Option<PathBuf>,
    pub games: usize,
    pub seed: String,
    pub max_steps: usize,
    pub opponent_policy: Option<String>,
    pub opponent_policy_mix_json: Option<String>,
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
}

impl EvalBridgeOptions {
    pub fn new(worker_script: impl Into<PathBuf>, genome_path: impl Into<PathBuf>) -> Self {
        Self {
            node_bin: "node".to_string(),
            worker_script: worker_script.into(),
            working_dir: None,
            genome_path: genome_path.into(),
            timeout_millis: None,
            opponent_genome_path: None,
            games: 3,
            seed: "neat-rust".to_string(),
            max_steps: 600,
            opponent_policy: None,
            opponent_policy_mix_json: None,
            first_turn_policy: "alternate".to_string(),
            fixed_first_turn: "human".to_string(),
            continuous_series: true,
            fitness_gold_scale: 1500.0,
            fitness_gold_neutral_delta: 0.0,
            fitness_win_weight: 0.85,
            fitness_gold_weight: 0.15,
            fitness_win_neutral_rate: 0.50,
            early_stop_win_rate_cutoffs_json: None,
            early_stop_go_take_rate_cutoffs_json: None,
            native_inference_backend: None,
        }
    }

    pub fn command_args(&self) -> Result<Vec<String>, EvalBridgeError> {
        let has_policy = self
            .opponent_policy
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_some();
        let has_mix = self
            .opponent_policy_mix_json
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_some();
        if !has_policy && !has_mix {
            return Err(EvalBridgeError::MissingOpponentPolicy);
        }

        validate_finite("--fitness-gold-scale", self.fitness_gold_scale)?;
        validate_finite(
            "--fitness-gold-neutral-delta",
            self.fitness_gold_neutral_delta,
        )?;
        validate_finite("--fitness-win-weight", self.fitness_win_weight)?;
        validate_finite("--fitness-gold-weight", self.fitness_gold_weight)?;
        validate_finite("--fitness-win-neutral-rate", self.fitness_win_neutral_rate)?;

        let mut args = Vec::new();
        args.push(path_arg(&self.worker_script));
        push_pair(&mut args, "--genome", path_arg(&self.genome_path));
        push_pair(&mut args, "--games", self.games.max(1).to_string());
        push_pair(&mut args, "--seed", self.seed.trim());
        push_pair(&mut args, "--max-steps", self.max_steps.max(20).to_string());

        if let Some(value) = non_empty_option(&self.opponent_policy) {
            push_pair(&mut args, "--opponent-policy", value);
        }
        if let Some(value) = non_empty_option(&self.opponent_policy_mix_json) {
            push_pair(&mut args, "--opponent-policy-mix", value);
        }
        if let Some(path) = &self.opponent_genome_path {
            push_pair(&mut args, "--opponent-genome", path_arg(path));
        }

        push_pair(
            &mut args,
            "--first-turn-policy",
            self.first_turn_policy.trim(),
        );
        push_pair(
            &mut args,
            "--fixed-first-turn",
            self.fixed_first_turn.trim(),
        );
        push_pair(
            &mut args,
            "--continuous-series",
            if self.continuous_series { "1" } else { "0" },
        );
        push_pair(
            &mut args,
            "--fitness-gold-scale",
            self.fitness_gold_scale.to_string(),
        );
        push_pair(
            &mut args,
            "--fitness-gold-neutral-delta",
            self.fitness_gold_neutral_delta.to_string(),
        );
        push_pair(
            &mut args,
            "--fitness-win-weight",
            self.fitness_win_weight.to_string(),
        );
        push_pair(
            &mut args,
            "--fitness-gold-weight",
            self.fitness_gold_weight.to_string(),
        );
        push_pair(
            &mut args,
            "--fitness-win-neutral-rate",
            self.fitness_win_neutral_rate.to_string(),
        );

        if let Some(value) = non_empty_option(&self.early_stop_win_rate_cutoffs_json) {
            push_pair(&mut args, "--early-stop-win-rate-cutoffs", value);
        }
        if let Some(value) = non_empty_option(&self.early_stop_go_take_rate_cutoffs_json) {
            push_pair(&mut args, "--early-stop-go-take-rate-cutoffs", value);
        }
        if let Some(value) = non_empty_option(&self.native_inference_backend) {
            push_pair(&mut args, "--native-inference-backend", value);
        }

        Ok(args)
    }
}

#[derive(Debug, Clone)]
pub struct EvalBridgeOutput {
    pub status_code: Option<i32>,
    pub command_args: Vec<String>,
    pub stdout: String,
    pub stderr: String,
    pub summary_json: Option<String>,
    pub eval_ok: Option<bool>,
    pub fitness: Option<f64>,
    pub win_rate: Option<f64>,
    pub mean_gold_delta: Option<f64>,
    pub go_take_rate: Option<f64>,
    pub go_fail_rate: Option<f64>,
    pub go_opportunity_count: Option<usize>,
    pub go_count: Option<usize>,
    pub go_games: Option<usize>,
    pub go_rate: Option<f64>,
    pub imitation_weighted_score: Option<f64>,
    pub eval_time_ms: Option<f64>,
    pub games: Option<usize>,
    pub early_stop_triggered: Option<bool>,
    pub early_stop_reason: Option<String>,
    pub early_stop_cutoff_games: Option<usize>,
}

#[derive(Debug)]
pub enum EvalBridgeError {
    MissingOpponentPolicy,
    InvalidNumber {
        name: &'static str,
        value: f64,
    },
    CommandIo(std::io::Error),
    WorkerTimeout {
        output: EvalBridgeOutput,
        timeout_millis: u64,
    },
    WorkerFailed(EvalBridgeOutput),
}

impl fmt::Display for EvalBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingOpponentPolicy => {
                write!(f, "--opponent-policy or --opponent-policy-mix is required")
            }
            Self::InvalidNumber { name, value } => {
                write!(f, "{name} must be finite, got {value}")
            }
            Self::CommandIo(err) => write!(f, "failed to run JS eval worker: {err}"),
            Self::WorkerTimeout {
                output,
                timeout_millis,
            } => {
                let stderr = output.stderr.trim();
                if stderr.is_empty() {
                    write!(f, "JS eval worker timed out after {timeout_millis} ms")
                } else {
                    write!(
                        f,
                        "JS eval worker timed out after {timeout_millis} ms: {stderr}"
                    )
                }
            }
            Self::WorkerFailed(output) => {
                let stderr = output.stderr.trim();
                if stderr.is_empty() {
                    write!(
                        f,
                        "JS eval worker failed with status {:?}",
                        output.status_code
                    )
                } else {
                    write!(
                        f,
                        "JS eval worker failed with status {:?}: {stderr}",
                        output.status_code
                    )
                }
            }
        }
    }
}

impl Error for EvalBridgeError {}

pub fn default_node_bin() -> String {
    let Ok(current_dir) = std::env::current_dir() else {
        return "node".to_string();
    };
    let mut bases = Vec::new();
    bases.push(current_dir.clone());
    if let Some(parent) = current_dir.parent() {
        bases.push(parent.to_path_buf());
    }

    for base in bases {
        let tools_dir = base.join(".tools");
        let pinned = tools_dir.join("node-v24.14.1-win-x64").join("node.exe");
        if pinned.exists() {
            return pinned.to_string_lossy().to_string();
        }

        if let Ok(entries) = fs::read_dir(&tools_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if name.starts_with("node-") {
                    let node = path.join("node.exe");
                    if node.exists() {
                        return node.to_string_lossy().to_string();
                    }
                }
            }
        }
    }

    "node".to_string()
}

pub fn run_neat_eval_worker(
    options: &EvalBridgeOptions,
) -> Result<EvalBridgeOutput, EvalBridgeError> {
    let args = options.command_args()?;
    let mut command = Command::new(&options.node_bin);
    command.args(&args);
    if let Some(working_dir) = &options.working_dir {
        command.current_dir(working_dir);
    }
    let output = run_command_with_timeout(&mut command, options.timeout_millis)?;
    let result = build_eval_output(args, output.status_code, output.stdout, output.stderr);

    if output.timed_out {
        Err(EvalBridgeError::WorkerTimeout {
            output: result,
            timeout_millis: options.timeout_millis.unwrap_or(0),
        })
    } else if output.success {
        Ok(result)
    } else {
        Err(EvalBridgeError::WorkerFailed(result))
    }
}

fn build_eval_output(
    command_args: Vec<String>,
    status_code: Option<i32>,
    stdout: String,
    stderr: String,
) -> EvalBridgeOutput {
    let summary_json = last_json_line(&stdout).map(str::to_string);
    let eval_ok = summary_json
        .as_deref()
        .and_then(|json| extract_bool_field(json, "eval_ok"));
    let fitness = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "fitness"));
    let win_rate = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "win_rate"));
    let mean_gold_delta = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "mean_gold_delta"));
    let go_take_rate = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "go_take_rate"));
    let go_fail_rate = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "go_fail_rate"));
    let go_opportunity_count = summary_json
        .as_deref()
        .and_then(|json| extract_usize_field(json, "go_opportunity_count"));
    let go_count = summary_json
        .as_deref()
        .and_then(|json| extract_usize_field(json, "go_count"));
    let go_games = summary_json
        .as_deref()
        .and_then(|json| extract_usize_field(json, "go_games"));
    let go_rate = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "go_rate"));
    let imitation_weighted_score = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "imitation_weighted_score"));
    let eval_time_ms = summary_json
        .as_deref()
        .and_then(|json| extract_number_field(json, "eval_time_ms"));
    let games = summary_json
        .as_deref()
        .and_then(|json| extract_usize_field(json, "games"));
    let early_stop_triggered = summary_json
        .as_deref()
        .and_then(|json| extract_bool_field(json, "early_stop_triggered"));
    let early_stop_reason = summary_json
        .as_deref()
        .and_then(|json| extract_string_field(json, "early_stop_reason"));
    let early_stop_cutoff_games = summary_json
        .as_deref()
        .and_then(|json| extract_usize_field(json, "early_stop_cutoff_games"));

    EvalBridgeOutput {
        status_code,
        command_args,
        stdout,
        stderr,
        summary_json,
        eval_ok,
        fitness,
        win_rate,
        mean_gold_delta,
        go_take_rate,
        go_fail_rate,
        go_opportunity_count,
        go_count,
        go_games,
        go_rate,
        imitation_weighted_score,
        eval_time_ms,
        games,
        early_stop_triggered,
        early_stop_reason,
        early_stop_cutoff_games,
    }
}

#[derive(Debug)]
struct CommandOutput {
    status_code: Option<i32>,
    success: bool,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

fn run_command_with_timeout(
    command: &mut Command,
    timeout_millis: Option<u64>,
) -> Result<CommandOutput, EvalBridgeError> {
    let timeout = timeout_millis
        .filter(|timeout| *timeout > 0)
        .map(Duration::from_millis);
    if timeout.is_none() {
        let output = command.output().map_err(EvalBridgeError::CommandIo)?;
        return Ok(CommandOutput {
            status_code: output.status.code(),
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            timed_out: false,
        });
    }

    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    let mut child = command.spawn().map_err(EvalBridgeError::CommandIo)?;
    let stdout_reader = spawn_pipe_reader(
        child
            .stdout
            .take()
            .ok_or_else(|| command_capture_error("stdout"))
            .map_err(EvalBridgeError::CommandIo)?,
    );
    let stderr_reader = spawn_pipe_reader(
        child
            .stderr
            .take()
            .ok_or_else(|| command_capture_error("stderr"))
            .map_err(EvalBridgeError::CommandIo)?,
    );

    let timeout = timeout.expect("timeout must exist after early return");
    let started_at = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().map_err(EvalBridgeError::CommandIo)? {
            break status;
        }
        if started_at.elapsed() >= timeout {
            timed_out = true;
            match child.kill() {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::InvalidInput => {}
                Err(err) => return Err(EvalBridgeError::CommandIo(err)),
            }
            break child.wait().map_err(EvalBridgeError::CommandIo)?;
        }
        thread::sleep(Duration::from_millis(10));
    };

    Ok(CommandOutput {
        status_code: status.code(),
        success: status.success(),
        stdout: join_pipe_reader(stdout_reader)?,
        stderr: join_pipe_reader(stderr_reader)?,
        timed_out,
    })
}

fn spawn_pipe_reader<R>(mut reader: R) -> thread::JoinHandle<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_pipe_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
) -> Result<String, EvalBridgeError> {
    let bytes = reader
        .join()
        .map_err(|_| EvalBridgeError::CommandIo(command_capture_error("reader thread panicked")))?
        .map_err(EvalBridgeError::CommandIo)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn command_capture_error(message: &str) -> io::Error {
    io::Error::other(message.to_string())
}

fn push_pair(args: &mut Vec<String>, key: impl Into<String>, value: impl ToString) {
    args.push(key.into());
    args.push(value.to_string());
}

fn path_arg(path: &std::path::Path) -> String {
    let value = path.to_string_lossy();
    let value = value.as_ref();
    value.strip_prefix(r"\\?\").unwrap_or(value).to_string()
}

fn non_empty_option(value: &Option<String>) -> Option<&str> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn validate_finite(name: &'static str, value: f64) -> Result<(), EvalBridgeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvalBridgeError::InvalidNumber { name, value })
    }
}

fn last_json_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with('{') && line.ends_with('}'))
}

fn extract_bool_field(json: &str, key: &str) -> Option<bool> {
    let value = field_value_start(json, key)?;
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_usize_field(json: &str, key: &str) -> Option<usize> {
    let value = extract_number_field(json, key)?;
    if value.is_finite() && value >= 0.0 {
        Some(value as usize)
    } else {
        None
    }
}

fn extract_string_field(json: &str, key: &str) -> Option<String> {
    let value = field_value_start(json, key)?;
    if value.starts_with("null") {
        return None;
    }
    if !value.starts_with('"') {
        return None;
    }
    parse_json_string_at(value, 0).map(|(value, _)| value)
}

fn extract_number_field(json: &str, key: &str) -> Option<f64> {
    let value = field_value_start(json, key)?;
    let end = value
        .char_indices()
        .take_while(|(_, ch)| matches!(ch, '0'..='9' | '-' | '+' | '.' | 'e' | 'E'))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if end == 0 {
        return None;
    }
    value[..end].parse::<f64>().ok()
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn eval_worker_timeout_returns_distinct_error() {
        let dir =
            std::env::temp_dir().join(format!("neat_rust_eval_bridge_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir should be created");

        let genome_path = dir.join("genome.json");
        fs::write(&genome_path, "{}").expect("genome json should be written");

        let worker_path = dir.join("slow_worker.mjs");
        fs::write(
            &worker_path,
            "setTimeout(() => { process.stdout.write('{\"eval_ok\":true,\"fitness\":1}\\n'); }, 300);\n",
        )
        .expect("worker script should be written");

        let mut options = EvalBridgeOptions::new(&worker_path, &genome_path);
        options.node_bin = default_node_bin();
        options.opponent_policy = Some("dummy".to_string());
        options.timeout_millis = Some(50);

        let err = run_neat_eval_worker(&options).expect_err("worker should time out");
        match err {
            EvalBridgeError::WorkerTimeout {
                output,
                timeout_millis,
            } => {
                assert_eq!(timeout_millis, 50);
                assert!(output.summary_json.is_none());
            }
            other => panic!("unexpected error: {other}"),
        }

        let _ = fs::remove_dir_all(&dir);
    }
}

fn field_value_start<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    let colon = after_key.find(':')?;
    Some(after_key[colon + 1..].trim_start())
}
