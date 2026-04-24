use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;
use crate::attributes::{BoolAttribute, ChoiceAttribute, FloatAttribute, RandomSource};
use crate::config::GenomeConfig;

use super::key::NodeKey;
use super::util::choose_copy;
use super::GeneError;

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultNodeGene {
    pub key: NodeKey,
    pub bias: f64,
    pub response: f64,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub time_constant: f64,
    pub iz_a: f64,
    pub iz_b: f64,
    pub iz_c: f64,
    pub iz_d: f64,
    pub memory_gate_enabled: bool,
    pub memory_gate_bias: f64,
    pub memory_gate_response: f64,
}

impl DefaultNodeGene {
    pub fn new(key: NodeKey) -> Self {
        Self {
            key,
            bias: 0.0,
            response: 1.0,
            activation: ActivationFunction::Sigmoid,
            aggregation: AggregationFunction::Sum,
            time_constant: 1.0,
            iz_a: 0.02,
            iz_b: 0.20,
            iz_c: -65.0,
            iz_d: 8.0,
            memory_gate_enabled: false,
            memory_gate_bias: 0.0,
            memory_gate_response: 1.0,
        }
    }

    pub fn initialized(
        key: NodeKey,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Self, GeneError> {
        let mut gene = Self::new(key);
        gene.init_attributes(config, rng)?;
        Ok(gene)
    }

    pub fn init_attributes(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GeneError> {
        self.bias = FloatAttribute::init_value(&config.bias, rng)?;
        self.response = FloatAttribute::init_value(&config.response, rng)?;
        self.activation = ChoiceAttribute::init_value(&config.activation, rng)?;
        self.aggregation = ChoiceAttribute::init_value(&config.aggregation, rng)?;
        self.time_constant = FloatAttribute::init_value(&config.time_constant, rng)?;
        self.iz_a = FloatAttribute::init_value(&config.iz_a, rng)?;
        self.iz_b = FloatAttribute::init_value(&config.iz_b, rng)?;
        self.iz_c = FloatAttribute::init_value(&config.iz_c, rng)?;
        self.iz_d = FloatAttribute::init_value(&config.iz_d, rng)?;
        self.memory_gate_enabled = BoolAttribute::init_value(&config.memory_gate_enabled, rng);
        self.memory_gate_bias = FloatAttribute::init_value(&config.memory_gate_bias, rng)?;
        self.memory_gate_response = FloatAttribute::init_value(&config.memory_gate_response, rng)?;
        Ok(())
    }

    pub fn mutate(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GeneError> {
        self.bias = FloatAttribute::mutate_value(self.bias, &config.bias, rng)?;
        self.response = FloatAttribute::mutate_value(self.response, &config.response, rng)?;
        self.activation = ChoiceAttribute::mutate_value(self.activation, &config.activation, rng)?;
        self.aggregation =
            ChoiceAttribute::mutate_value(self.aggregation, &config.aggregation, rng)?;
        self.time_constant =
            FloatAttribute::mutate_value(self.time_constant, &config.time_constant, rng)?;
        self.iz_a = FloatAttribute::mutate_value(self.iz_a, &config.iz_a, rng)?;
        self.iz_b = FloatAttribute::mutate_value(self.iz_b, &config.iz_b, rng)?;
        self.iz_c = FloatAttribute::mutate_value(self.iz_c, &config.iz_c, rng)?;
        self.iz_d = FloatAttribute::mutate_value(self.iz_d, &config.iz_d, rng)?;
        self.memory_gate_enabled =
            BoolAttribute::mutate_value(self.memory_gate_enabled, &config.memory_gate_enabled, rng);
        self.memory_gate_bias =
            FloatAttribute::mutate_value(self.memory_gate_bias, &config.memory_gate_bias, rng)?;
        self.memory_gate_response = FloatAttribute::mutate_value(
            self.memory_gate_response,
            &config.memory_gate_response,
            rng,
        )?;
        Ok(())
    }

    pub fn crossover(&self, other: &Self, rng: &mut impl RandomSource) -> Result<Self, GeneError> {
        if self.key != other.key {
            return Err(GeneError::KeyMismatch {
                left: self.key.to_string(),
                right: other.key.to_string(),
            });
        }

        Ok(Self {
            key: self.key,
            bias: choose_copy(self.bias, other.bias, rng),
            response: choose_copy(self.response, other.response, rng),
            activation: choose_copy(self.activation, other.activation, rng),
            aggregation: choose_copy(self.aggregation, other.aggregation, rng),
            time_constant: choose_copy(self.time_constant, other.time_constant, rng),
            iz_a: choose_copy(self.iz_a, other.iz_a, rng),
            iz_b: choose_copy(self.iz_b, other.iz_b, rng),
            iz_c: choose_copy(self.iz_c, other.iz_c, rng),
            iz_d: choose_copy(self.iz_d, other.iz_d, rng),
            memory_gate_enabled: choose_copy(
                self.memory_gate_enabled,
                other.memory_gate_enabled,
                rng,
            ),
            memory_gate_bias: choose_copy(self.memory_gate_bias, other.memory_gate_bias, rng),
            memory_gate_response: choose_copy(
                self.memory_gate_response,
                other.memory_gate_response,
                rng,
            ),
        })
    }

    pub fn distance(&self, other: &Self, config: &GenomeConfig) -> Result<f64, GeneError> {
        if self.key != other.key {
            return Err(GeneError::KeyMismatch {
                left: self.key.to_string(),
                right: other.key.to_string(),
            });
        }

        let mut distance = (self.bias - other.bias).abs();
        distance += (self.response - other.response).abs();
        distance += (self.time_constant - other.time_constant).abs();
        distance += (self.iz_a - other.iz_a).abs();
        distance += (self.iz_b - other.iz_b).abs();
        distance += (self.iz_c - other.iz_c).abs();
        distance += (self.iz_d - other.iz_d).abs();
        if self.memory_gate_enabled != other.memory_gate_enabled {
            distance += config.compatibility_enable_penalty;
        }
        distance += (self.memory_gate_bias - other.memory_gate_bias).abs();
        distance += (self.memory_gate_response - other.memory_gate_response).abs();
        if self.activation != other.activation {
            distance += 1.0;
        }
        if self.aggregation != other.aggregation {
            distance += 1.0;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }
}
