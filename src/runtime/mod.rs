//! Application runtime entry points built on top of the core and compatibility layers.

pub mod kflower {
    pub use crate::train_runner::{
        run_kflower_training, TrainEvalBackend, TrainRunSummary, TrainRunnerError,
        TrainRunnerOptions, TrainRuntimeConfig,
    };
}
