use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use super::EvalBridgeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCommand(String);

impl NodeCommand {
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            Self("node".to_string())
        } else {
            Self(trimmed.to_string())
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for NodeCommand {
    fn default() -> Self {
        Self::new("node")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalSeed(String);

impl EvalSeed {
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            Self::default()
        } else {
            Self(trimmed.to_string())
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for EvalSeed {
    fn default() -> Self {
        Self("neat-rust".to_string())
    }
}

macro_rules! positive_count {
    ($name:ident, $min:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(NonZeroUsize);

        impl $name {
            pub fn new(value: usize) -> Self {
                Self(
                    NonZeroUsize::new(value.max($min))
                        .expect("positive count is clamped to at least one"),
                )
            }

            pub fn get(self) -> usize {
                self.0.get()
            }
        }
    };
}

positive_count!(BridgeGameCount, 1);
positive_count!(BridgeStepCount, 20);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeJsonArrayArg(String);

impl BridgeJsonArrayArg {
    pub fn optional(
        field_name: &'static str,
        value: Option<String>,
    ) -> Result<Option<Self>, EvalBridgeError> {
        let Some(value) = value else {
            return Ok(None);
        };
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let parsed: Value =
            serde_json::from_str(trimmed).map_err(|err| EvalBridgeError::InvalidJsonArray {
                name: field_name,
                message: err.to_string(),
            })?;
        let Some(items) = parsed.as_array() else {
            return Err(EvalBridgeError::InvalidJsonArray {
                name: field_name,
                message: "expected JSON array".to_string(),
            });
        };
        if items.is_empty() {
            Ok(None)
        } else {
            serde_json::to_string(&parsed)
                .map(Self)
                .map(Some)
                .map_err(|err| EvalBridgeError::InvalidJsonArray {
                    name: field_name,
                    message: err.to_string(),
                })
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeOpponent {
    None,
    Policy(String),
    Mix(BridgeJsonArrayArg),
    PolicyWithMix {
        policy: String,
        mix: BridgeJsonArrayArg,
    },
}

impl BridgeOpponent {
    pub fn from_parts(
        policy: Option<String>,
        mix_json: Option<String>,
    ) -> Result<Self, EvalBridgeError> {
        let policy = policy
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let mix = BridgeJsonArrayArg::optional("--opponent-policy-mix", mix_json)?;
        Ok(match (policy, mix) {
            (Some(policy), Some(mix)) => Self::PolicyWithMix { policy, mix },
            (Some(policy), None) => Self::Policy(policy),
            (None, Some(mix)) => Self::Mix(mix),
            (None, None) => Self::None,
        })
    }

    pub fn policy_arg(&self) -> Option<&str> {
        match self {
            Self::Policy(policy) | Self::PolicyWithMix { policy, .. } => Some(policy.as_str()),
            Self::None | Self::Mix(_) => None,
        }
    }

    pub fn mix_arg(&self) -> Option<&str> {
        match self {
            Self::Mix(mix) | Self::PolicyWithMix { mix, .. } => Some(mix.as_str()),
            Self::None | Self::Policy(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }
}

impl Default for BridgeOpponent {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeSeat {
    Ai,
    Human,
}

impl BridgeSeat {
    fn parse(value: &str) -> Result<Self, EvalBridgeError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ai" | "model" | "agent" | "first" => Ok(Self::Ai),
            "human" | "opponent" | "heuristic" | "second" => Ok(Self::Human),
            other => Err(EvalBridgeError::InvalidTurnPolicy(format!(
                "unsupported fixed first turn {other:?}; expected ai or human"
            ))),
        }
    }

    fn as_arg(self) -> &'static str {
        match self {
            Self::Ai => "ai",
            Self::Human => "human",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeTurnPolicy {
    Alternate,
    Fixed(BridgeSeat),
}

impl BridgeTurnPolicy {
    pub fn from_parts(
        policy: Option<String>,
        fixed_first_turn: Option<String>,
    ) -> Result<Self, EvalBridgeError> {
        let policy = policy
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty());
        match policy.as_deref() {
            None | Some("alternate" | "alternating" | "switch" | "switch-seats") => {
                Ok(Self::Alternate)
            }
            Some("fixed" | "fixed-seat") => {
                let seat = fixed_first_turn
                    .as_deref()
                    .map(BridgeSeat::parse)
                    .transpose()?
                    .unwrap_or(BridgeSeat::Human);
                Ok(Self::Fixed(seat))
            }
            Some("ai" | "model" | "first") => Ok(Self::Fixed(BridgeSeat::Ai)),
            Some("human" | "opponent" | "second") => Ok(Self::Fixed(BridgeSeat::Human)),
            Some(other) => Err(EvalBridgeError::InvalidTurnPolicy(format!(
                "unsupported first turn policy {other:?}; expected alternate or fixed"
            ))),
        }
    }

    pub fn as_policy_arg(self) -> &'static str {
        match self {
            Self::Alternate => "alternate",
            Self::Fixed(_) => "fixed",
        }
    }

    pub fn fixed_first_turn_arg(self) -> &'static str {
        match self {
            Self::Alternate => BridgeSeat::Human.as_arg(),
            Self::Fixed(seat) => seat.as_arg(),
        }
    }
}

impl Default for BridgeTurnPolicy {
    fn default() -> Self {
        Self::Alternate
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BridgeEarlyStopConfig {
    pub win_rate_cutoffs: Option<BridgeJsonArrayArg>,
    pub go_take_rate_cutoffs: Option<BridgeJsonArrayArg>,
}

impl BridgeEarlyStopConfig {
    pub fn from_parts(
        win_rate_cutoffs: Option<String>,
        go_take_rate_cutoffs: Option<String>,
    ) -> Result<Self, EvalBridgeError> {
        Ok(Self {
            win_rate_cutoffs: BridgeJsonArrayArg::optional(
                "--early-stop-win-rate-cutoffs",
                win_rate_cutoffs,
            )?,
            go_take_rate_cutoffs: BridgeJsonArrayArg::optional(
                "--early-stop-go-take-rate-cutoffs",
                go_take_rate_cutoffs,
            )?,
        })
    }

    pub fn win_rate_arg(&self) -> Option<&str> {
        self.win_rate_cutoffs
            .as_ref()
            .map(BridgeJsonArrayArg::as_str)
    }

    pub fn go_take_rate_arg(&self) -> Option<&str> {
        self.go_take_rate_cutoffs
            .as_ref()
            .map(BridgeJsonArrayArg::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeNativeInferenceBackend {
    Off,
    Named(String),
}

impl BridgeNativeInferenceBackend {
    pub fn from_optional(value: Option<String>) -> Self {
        let Some(value) = value else {
            return Self::Off;
        };
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "" | "off" | "none" | "false" | "0" => Self::Off,
            _ => Self::Named(normalized),
        }
    }

    pub fn as_arg(&self) -> Option<&str> {
        match self {
            Self::Off => None,
            Self::Named(value) => Some(value.as_str()),
        }
    }
}

impl Default for BridgeNativeInferenceBackend {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Debug, Clone)]
pub struct EvalBridgeOptions {
    pub node_bin: NodeCommand,
    pub worker_script: PathBuf,
    pub working_dir: Option<PathBuf>,
    pub genome_path: PathBuf,
    pub timeout: Option<Duration>,
    pub opponent_genome_path: Option<PathBuf>,
    pub games: BridgeGameCount,
    pub seed: EvalSeed,
    pub max_steps: BridgeStepCount,
    pub opponent: BridgeOpponent,
    pub turn_policy: BridgeTurnPolicy,
    pub continuous_series: bool,
    pub fitness_gold_scale: f64,
    pub fitness_gold_neutral_delta: f64,
    pub fitness_win_weight: f64,
    pub fitness_gold_weight: f64,
    pub fitness_win_neutral_rate: f64,
    pub early_stop: BridgeEarlyStopConfig,
    pub native_inference_backend: BridgeNativeInferenceBackend,
}

impl EvalBridgeOptions {
    pub fn new(worker_script: impl Into<PathBuf>, genome_path: impl Into<PathBuf>) -> Self {
        Self {
            node_bin: NodeCommand::default(),
            worker_script: worker_script.into(),
            working_dir: None,
            genome_path: genome_path.into(),
            timeout: None,
            opponent_genome_path: None,
            games: BridgeGameCount::new(3),
            seed: EvalSeed::default(),
            max_steps: BridgeStepCount::new(600),
            opponent: BridgeOpponent::default(),
            turn_policy: BridgeTurnPolicy::default(),
            continuous_series: true,
            fitness_gold_scale: 1500.0,
            fitness_gold_neutral_delta: 0.0,
            fitness_win_weight: 0.85,
            fitness_gold_weight: 0.15,
            fitness_win_neutral_rate: 0.50,
            early_stop: BridgeEarlyStopConfig::default(),
            native_inference_backend: BridgeNativeInferenceBackend::default(),
        }
    }

    pub fn command_args(&self) -> Result<Vec<String>, EvalBridgeError> {
        if self.opponent.is_empty() {
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
        push_pair(&mut args, "--games", self.games.get().to_string());
        push_pair(&mut args, "--seed", self.seed.as_str());
        push_pair(&mut args, "--max-steps", self.max_steps.get().to_string());

        if let Some(value) = self.opponent.policy_arg() {
            push_pair(&mut args, "--opponent-policy", value);
        }
        if let Some(value) = self.opponent.mix_arg() {
            push_pair(&mut args, "--opponent-policy-mix", value);
        }
        if let Some(path) = &self.opponent_genome_path {
            push_pair(&mut args, "--opponent-genome", path_arg(path));
        }

        push_pair(
            &mut args,
            "--first-turn-policy",
            self.turn_policy.as_policy_arg(),
        );
        push_pair(
            &mut args,
            "--fixed-first-turn",
            self.turn_policy.fixed_first_turn_arg(),
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

        if let Some(value) = self.early_stop.win_rate_arg() {
            push_pair(&mut args, "--early-stop-win-rate-cutoffs", value);
        }
        if let Some(value) = self.early_stop.go_take_rate_arg() {
            push_pair(&mut args, "--early-stop-go-take-rate-cutoffs", value);
        }
        if let Some(value) = self.native_inference_backend.as_arg() {
            push_pair(&mut args, "--native-inference-backend", value);
        }

        Ok(args)
    }
}

fn push_pair(args: &mut Vec<String>, key: impl Into<String>, value: impl ToString) {
    args.push(key.into());
    args.push(value.to_string());
}

fn path_arg(path: &Path) -> String {
    let value = path.to_string_lossy();
    let value = value.as_ref();
    value.strip_prefix(r"\\?\").unwrap_or(value).to_string()
}

fn validate_finite(name: &'static str, value: f64) -> Result<(), EvalBridgeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvalBridgeError::InvalidNumber { name, value })
    }
}
