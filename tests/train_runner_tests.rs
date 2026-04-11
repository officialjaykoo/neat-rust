use neat_rust::{TrainEvalBackend, TrainRunnerError, TrainRuntimeConfig};

#[test]
fn runtime_config_parses_eval_timeout_sec_for_worker_hardening() {
    let runtime = TrainRuntimeConfig::from_json_text(
        r#"{
            "generations": 3,
            "eval_workers": 2,
            "games_per_genome": 5,
            "eval_timeout_sec": 360,
            "max_eval_steps": 600,
            "checkpoint_every": 1,
            "opponent_policy": "H-CL"
        }"#,
    )
    .expect("runtime config should parse");

    assert_eq!(runtime.eval_timeout_sec, 360);
    assert_eq!(runtime.eval_workers, 2);
    assert_eq!(runtime.eval_backend, TrainEvalBackend::JsWorker);
    assert_eq!(
        runtime.winner_playoff_eval_backend,
        TrainEvalBackend::JsWorker
    );
}

#[test]
fn runtime_config_normalizes_eval_backend_aliases() {
    let runtime = TrainRuntimeConfig::from_json_text(
        r#"{
            "eval_backend": "cuda",
            "winner_playoff_eval_backend": "js"
        }"#,
    )
    .expect("runtime config should parse");

    assert_eq!(runtime.eval_backend, TrainEvalBackend::CudaNative);
    assert_eq!(
        runtime.winner_playoff_eval_backend,
        TrainEvalBackend::JsWorker
    );
}

#[test]
fn runtime_config_rejects_cuda_backend_for_current_kflower_path() {
    let runtime = TrainRuntimeConfig::from_json_text(
        r#"{
            "eval_backend": "cuda_native"
        }"#,
    )
    .expect("runtime config should parse");

    let err = runtime
        .validate()
        .expect_err("cuda backend should be rejected for current train runner");
    match err {
        TrainRunnerError::RuntimeConfig(message) => {
            assert!(message.contains("scripts/neat_eval_worker.mjs"));
            assert!(message.contains("GPUCTRNNEvaluator"));
        }
        other => panic!("unexpected error: {other}"),
    }
}
