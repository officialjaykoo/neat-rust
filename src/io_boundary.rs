//! NEAT TOML config, genome export, and checkpoint boundary.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::checkpoint::{CheckpointError, Checkpointer};
use crate::config::{Config, ConfigError};
use crate::export::{export_genome_json, GenomeJsonOptions};

use crate::evolution::{PopulationCheckpointError, PopulationCheckpointSink};
use crate::genome::DefaultGenome;
use crate::population::Population;

#[derive(Debug, Clone, PartialEq)]
pub struct RustCheckpointSink {
    checkpointer: Checkpointer,
}

impl RustCheckpointSink {
    pub fn new(
        generation_interval: Option<usize>,
        filename_prefix: impl Into<String>,
        config_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            checkpointer: Checkpointer::new(generation_interval, filename_prefix)
                .with_config_path(config_path),
        }
    }

    pub fn checkpointer(&self) -> &Checkpointer {
        &self.checkpointer
    }
}

impl PopulationCheckpointSink for RustCheckpointSink {
    fn should_save(&self, generation: usize) -> bool {
        self.checkpointer.should_save(generation)
    }

    fn checkpoint_path(&self, generation: usize) -> PathBuf {
        self.checkpointer.checkpoint_path(generation)
    }

    fn save_population(
        &self,
        population: &Population,
    ) -> Result<PathBuf, PopulationCheckpointError> {
        self.checkpointer
            .save_checkpoint(population)
            .map_err(|err| PopulationCheckpointError::new(err.to_string()))
    }
}

pub fn load_neat_config(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
    Config::from_file(path)
}

pub fn export_neat_genome_json(
    genome: &DefaultGenome,
    config: &Config,
    feature_profile: impl AsRef<str>,
) -> String {
    export_genome_json(
        genome,
        &config.genome,
        &GenomeJsonOptions::new(feature_profile.as_ref()),
    )
}

pub fn new_rust_checkpointer(
    generation_interval: Option<usize>,
    filename_prefix: impl Into<String>,
    config_path: impl Into<PathBuf>,
) -> Checkpointer {
    RustCheckpointSink::new(generation_interval, filename_prefix, config_path)
        .checkpointer
        .clone()
}

pub fn new_rust_checkpoint_sink(
    generation_interval: Option<usize>,
    filename_prefix: impl Into<String>,
    config_path: impl Into<PathBuf>,
) -> Arc<dyn PopulationCheckpointSink> {
    Arc::new(RustCheckpointSink::new(
        generation_interval,
        filename_prefix,
        config_path,
    ))
}

pub fn save_rust_checkpoint(
    checkpointer: &Checkpointer,
    population: &Population,
) -> Result<PathBuf, CheckpointError> {
    checkpointer.save_checkpoint(population)
}

pub fn restore_rust_checkpoint(path: impl AsRef<Path>) -> Result<Population, CheckpointError> {
    Checkpointer::restore_checkpoint(path)
}
