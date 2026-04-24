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
    pub enabled: bool,
}

impl DefaultConnectionGene {
    pub fn new(key: ConnectionKey) -> Self {
        Self {
            key,
            innovation: None,
            weight: 0.0,
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
        self.enabled = BoolAttribute::init_value(&config.enabled, rng);
        Ok(())
    }

    pub fn mutate(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GeneError> {
        self.weight = FloatAttribute::mutate_value(self.weight, &config.weight, rng)?;
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
        let mut enabled = choose_copy(self.enabled, other.enabled, rng);
        if !self.enabled || !other.enabled {
            enabled = rng.next_f64() >= 0.75;
        }

        Ok(Self {
            key: self.key,
            innovation: self.innovation.or(other.innovation),
            weight,
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
        if self.enabled != other.enabled {
            distance += config.compatibility_enable_penalty;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }
}
