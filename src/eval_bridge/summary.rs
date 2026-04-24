use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub(super) struct EvalWorkerSummary {
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

impl EvalWorkerSummary {
    pub(super) fn parse(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }
}

pub(super) fn last_json_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with('{') && line.ends_with('}'))
}
