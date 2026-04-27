use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

mod options;
mod summary;

pub use options::{
    BridgeEarlyStopConfig, BridgeGameCount, BridgeJsonArrayArg, BridgeNativeInferenceBackend,
    BridgeOpponent, BridgeSeat, BridgeStepCount, BridgeTurnPolicy, EvalBridgeOptions, EvalSeed,
    ExternalEvalCommand,
};
use summary::{last_json_line, EvalWorkerSummary};

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
    InvalidJsonArray {
        name: &'static str,
        message: String,
    },
    InvalidTurnPolicy(String),
    CommandIo(std::io::Error),
    WorkerTimeout {
        output: Box<EvalBridgeOutput>,
        timeout_millis: u64,
    },
    WorkerFailed(Box<EvalBridgeOutput>),
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
            Self::InvalidJsonArray { name, message } => {
                write!(f, "{name} must be a JSON array: {message}")
            }
            Self::InvalidTurnPolicy(message) => write!(f, "{message}"),
            Self::CommandIo(err) => write!(f, "failed to run external eval worker: {err}"),
            Self::WorkerTimeout {
                output,
                timeout_millis,
            } => {
                let stderr = output.stderr.trim();
                if stderr.is_empty() {
                    write!(
                        f,
                        "external eval worker timed out after {timeout_millis} ms"
                    )
                } else {
                    write!(
                        f,
                        "external eval worker timed out after {timeout_millis} ms: {stderr}"
                    )
                }
            }
            Self::WorkerFailed(output) => {
                let stderr = output.stderr.trim();
                if stderr.is_empty() {
                    write!(
                        f,
                        "external eval worker failed with status {:?}",
                        output.status_code
                    )
                } else {
                    write!(
                        f,
                        "external eval worker failed with status {:?}: {stderr}",
                        output.status_code
                    )
                }
            }
        }
    }
}

impl Error for EvalBridgeError {}

pub fn default_external_eval_command() -> String {
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

pub fn run_external_eval_worker(
    options: &EvalBridgeOptions,
) -> Result<EvalBridgeOutput, EvalBridgeError> {
    let args = options.command_args()?;
    let mut command = Command::new(options.worker_command.as_str());
    command.args(&args);
    if let Some(working_dir) = &options.working_dir {
        command.current_dir(working_dir);
    }
    let output = run_command_with_timeout(&mut command, options.timeout)?;
    let result = build_eval_output(args, output.status_code, output.stdout, output.stderr);

    if output.timed_out {
        Err(EvalBridgeError::WorkerTimeout {
            output: Box::new(result),
            timeout_millis: timeout_millis(options.timeout),
        })
    } else if output.success {
        Ok(result)
    } else {
        Err(EvalBridgeError::WorkerFailed(Box::new(result)))
    }
}

fn build_eval_output(
    command_args: Vec<String>,
    status_code: Option<i32>,
    stdout: String,
    stderr: String,
) -> EvalBridgeOutput {
    let summary_json = last_json_line(&stdout).map(str::to_string);
    let summary = summary_json
        .as_deref()
        .and_then(EvalWorkerSummary::parse)
        .unwrap_or_default();

    EvalBridgeOutput {
        status_code,
        command_args,
        stdout,
        stderr,
        summary_json,
        eval_ok: summary.eval_ok,
        fitness: summary.fitness,
        win_rate: summary.win_rate,
        mean_gold_delta: summary.mean_gold_delta,
        go_take_rate: summary.go_take_rate,
        go_fail_rate: summary.go_fail_rate,
        go_opportunity_count: summary.go_opportunity_count,
        go_count: summary.go_count,
        go_games: summary.go_games,
        go_rate: summary.go_rate,
        imitation_weighted_score: summary.imitation_weighted_score,
        eval_time_ms: summary.eval_time_ms,
        games: summary.games,
        early_stop_triggered: summary.early_stop_triggered,
        early_stop_reason: summary.early_stop_reason,
        early_stop_cutoff_games: summary.early_stop_cutoff_games,
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
    timeout: Option<Duration>,
) -> Result<CommandOutput, EvalBridgeError> {
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

    let Some(timeout) = timeout else {
        unreachable!("timeout was handled by early return");
    };
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

fn timeout_millis(timeout: Option<Duration>) -> u64 {
    timeout
        .map(|timeout| timeout.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn external_worker_timeout_returns_distinct_error() {
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
        options.worker_command = ExternalEvalCommand::new(default_external_eval_command());
        options.opponent = BridgeOpponent::from_parts(Some("dummy".to_string()), None)
            .expect("dummy opponent should parse");
        options.timeout = Some(Duration::from_millis(50));

        let err = run_external_eval_worker(&options).expect_err("worker should time out");
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

    #[test]
    fn external_worker_summary_uses_json_parser() {
        let summary = EvalWorkerSummary::parse(
            r#"{"eval_ok":true,"fitness":1.25,"games":12,"early_stop_reason":"low win rate"}"#,
        )
        .expect("summary json should parse");
        assert_eq!(summary.eval_ok, Some(true));
        assert_eq!(summary.fitness, Some(1.25));
        assert_eq!(summary.games, Some(12));
        assert_eq!(summary.early_stop_reason.as_deref(), Some("low win rate"));
    }
}
