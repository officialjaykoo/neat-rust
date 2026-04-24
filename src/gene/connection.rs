use crate::attributes::{BoolAttribute, FloatAttribute, RandomSource};
use crate::config::GenomeConfig;

use super::key::ConnectionKey;
use super::util::choose_copy;
use super::GeneError;

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultConnectionGene {
    pub key: ConnectionKey,
    pub innovation: Option<i64>,
    pub weight: f64,
    pub connection_gru_enabled: bool,
    pub connection_memory_weight: f64,
    pub connection_reset_input_weight: f64,
    pub connection_reset_memory_weight: f64,
    pub connection_update_input_weight: f64,
    pub connection_update_memory_weight: f64,
    pub enabled: bool,
}

impl DefaultConnectionGene {
    pub fn new(key: ConnectionKey) -> Self {
        Self {
            key,
            innovation: None,
            weight: 0.0,
            connection_gru_enabled: false,
            connection_memory_weight: 0.0,
            connection_reset_input_weight: 0.0,
            connection_reset_memory_weight: 0.0,
            connection_update_input_weight: 0.0,
            connection_update_memory_weight: 0.0,
            enabled: false,
        }
    }

    pub fn with_innovation(key: ConnectionKey, innovation: i64) -> Self {
        let mut gene = Self::new(key);
        gene.innovation = Some(innovation);
        gene
    }

    pub fn initialized(
        key: ConnectionKey,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Self, GeneError> {
        let mut gene = Self::new(key);
        gene.init_attributes(config, rng)?;
        Ok(gene)
    }

    pub fn initialized_with_innovation(
        key: ConnectionKey,
        innovation: i64,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Self, GeneError> {
        let mut gene = Self::with_innovation(key, innovation);
        gene.init_attributes(config, rng)?;
        Ok(gene)
    }

    pub fn init_attributes(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GeneError> {
        self.weight = FloatAttribute::init_value(&config.weight, rng)?;
        self.connection_gru_enabled =
            BoolAttribute::init_value(&config.connection_gru_enabled, rng);
        self.connection_memory_weight =
            FloatAttribute::init_value(&config.connection_memory_weight, rng)?;
        self.connection_reset_input_weight =
            FloatAttribute::init_value(&config.connection_reset_input_weight, rng)?;
        self.connection_reset_memory_weight =
            FloatAttribute::init_value(&config.connection_reset_memory_weight, rng)?;
        self.connection_update_input_weight =
            FloatAttribute::init_value(&config.connection_update_input_weight, rng)?;
        self.connection_update_memory_weight =
            FloatAttribute::init_value(&config.connection_update_memory_weight, rng)?;
        self.enabled = BoolAttribute::init_value(&config.enabled, rng);
        Ok(())
    }

    pub fn mutate(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GeneError> {
        self.weight = FloatAttribute::mutate_value(self.weight, &config.weight, rng)?;
        self.connection_gru_enabled = BoolAttribute::mutate_value(
            self.connection_gru_enabled,
            &config.connection_gru_enabled,
            rng,
        );
        self.connection_memory_weight = FloatAttribute::mutate_value(
            self.connection_memory_weight,
            &config.connection_memory_weight,
            rng,
        )?;
        self.connection_reset_input_weight = FloatAttribute::mutate_value(
            self.connection_reset_input_weight,
            &config.connection_reset_input_weight,
            rng,
        )?;
        self.connection_reset_memory_weight = FloatAttribute::mutate_value(
            self.connection_reset_memory_weight,
            &config.connection_reset_memory_weight,
            rng,
        )?;
        self.connection_update_input_weight = FloatAttribute::mutate_value(
            self.connection_update_input_weight,
            &config.connection_update_input_weight,
            rng,
        )?;
        self.connection_update_memory_weight = FloatAttribute::mutate_value(
            self.connection_update_memory_weight,
            &config.connection_update_memory_weight,
            rng,
        )?;
        self.enabled = BoolAttribute::mutate_value(self.enabled, &config.enabled, rng);
        Ok(())
    }

    pub fn crossover(&self, other: &Self, rng: &mut impl RandomSource) -> Result<Self, GeneError> {
        if self.key != other.key {
            return Err(GeneError::KeyMismatch {
                left: self.key.to_string(),
                right: other.key.to_string(),
            });
        }
        if let (Some(left), Some(right)) = (self.innovation, other.innovation) {
            if left != right {
                return Err(GeneError::InnovationMismatch { left, right });
            }
        }

        let weight = choose_copy(self.weight, other.weight, rng);
        let connection_gru_enabled = choose_copy(
            self.connection_gru_enabled,
            other.connection_gru_enabled,
            rng,
        );
        let connection_memory_weight = choose_copy(
            self.connection_memory_weight,
            other.connection_memory_weight,
            rng,
        );
        let connection_reset_input_weight = choose_copy(
            self.connection_reset_input_weight,
            other.connection_reset_input_weight,
            rng,
        );
        let connection_reset_memory_weight = choose_copy(
            self.connection_reset_memory_weight,
            other.connection_reset_memory_weight,
            rng,
        );
        let connection_update_input_weight = choose_copy(
            self.connection_update_input_weight,
            other.connection_update_input_weight,
            rng,
        );
        let connection_update_memory_weight = choose_copy(
            self.connection_update_memory_weight,
            other.connection_update_memory_weight,
            rng,
        );
        let mut enabled = choose_copy(self.enabled, other.enabled, rng);
        if !self.enabled || !other.enabled {
            enabled = rng.next_f64() >= 0.75;
        }

        Ok(Self {
            key: self.key,
            innovation: self.innovation.or(other.innovation),
            weight,
            connection_gru_enabled,
            connection_memory_weight,
            connection_reset_input_weight,
            connection_reset_memory_weight,
            connection_update_input_weight,
            connection_update_memory_weight,
            enabled,
        })
    }

    pub fn distance(&self, other: &Self, config: &GenomeConfig) -> Result<f64, GeneError> {
        if self.key != other.key {
            return Err(GeneError::KeyMismatch {
                left: self.key.to_string(),
                right: other.key.to_string(),
            });
        }

        let mut distance = (self.weight - other.weight).abs();
        if self.connection_gru_enabled != other.connection_gru_enabled {
            distance += config.compatibility_enable_penalty;
        }
        distance += (self.connection_memory_weight - other.connection_memory_weight).abs();
        distance +=
            (self.connection_reset_input_weight - other.connection_reset_input_weight).abs();
        distance +=
            (self.connection_reset_memory_weight - other.connection_reset_memory_weight).abs();
        distance +=
            (self.connection_update_input_weight - other.connection_update_input_weight).abs();
        distance +=
            (self.connection_update_memory_weight - other.connection_update_memory_weight).abs();
        if self.enabled != other.enabled {
            distance += config.compatibility_enable_penalty;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }
}
