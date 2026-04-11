use std::error::Error;
use std::fmt;

use crate::attributes::{
    AttributeError, BoolAttribute, FloatAttribute, RandomSource, StringAttribute,
};
use crate::config::GenomeConfig;

pub type NodeKey = i64;
pub type ConnectionKey = (NodeKey, NodeKey);

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultNodeGene {
    pub key: NodeKey,
    pub bias: f64,
    pub response: f64,
    pub activation: String,
    pub aggregation: String,
    pub time_constant: f64,
    pub iz_a: f64,
    pub iz_b: f64,
    pub iz_c: f64,
    pub iz_d: f64,
    pub memory_gate_enabled: bool,
    pub memory_gate_bias: f64,
    pub memory_gate_response: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultConnectionGene {
    pub key: ConnectionKey,
    pub innovation: Option<i64>,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeneError {
    Attribute(AttributeError),
    InnovationMismatch { left: i64, right: i64 },
    KeyMismatch { left: String, right: String },
}

impl fmt::Display for GeneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attribute(err) => write!(f, "{err}"),
            Self::InnovationMismatch { left, right } => {
                write!(f, "gene innovation mismatch: left={left}, right={right}")
            }
            Self::KeyMismatch { left, right } => {
                write!(f, "gene key mismatch: left={left}, right={right}")
            }
        }
    }
}

impl Error for GeneError {}

impl From<AttributeError> for GeneError {
    fn from(value: AttributeError) -> Self {
        Self::Attribute(value)
    }
}

impl DefaultNodeGene {
    pub fn new(key: NodeKey) -> Self {
        Self {
            key,
            bias: 0.0,
            response: 1.0,
            activation: "sigmoid".to_string(),
            aggregation: "sum".to_string(),
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
        self.activation = StringAttribute::init_value(&config.activation, rng)?;
        self.aggregation = StringAttribute::init_value(&config.aggregation, rng)?;
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
        self.activation = StringAttribute::mutate_value(&self.activation, &config.activation, rng)?;
        self.aggregation =
            StringAttribute::mutate_value(&self.aggregation, &config.aggregation, rng)?;
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
            bias: choose_f64(self.bias, other.bias, rng),
            response: choose_f64(self.response, other.response, rng),
            activation: choose_string(&self.activation, &other.activation, rng),
            aggregation: choose_string(&self.aggregation, &other.aggregation, rng),
            time_constant: choose_f64(self.time_constant, other.time_constant, rng),
            iz_a: choose_f64(self.iz_a, other.iz_a, rng),
            iz_b: choose_f64(self.iz_b, other.iz_b, rng),
            iz_c: choose_f64(self.iz_c, other.iz_c, rng),
            iz_d: choose_f64(self.iz_d, other.iz_d, rng),
            memory_gate_enabled: choose_bool(
                self.memory_gate_enabled,
                other.memory_gate_enabled,
                rng,
            ),
            memory_gate_bias: choose_f64(self.memory_gate_bias, other.memory_gate_bias, rng),
            memory_gate_response: choose_f64(
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
        if self.activation != other.activation {
            distance += 1.0;
        }
        if self.aggregation != other.aggregation {
            distance += 1.0;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }
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
                left: format!("{:?}", self.key),
                right: format!("{:?}", other.key),
            });
        }
        if let (Some(left), Some(right)) = (self.innovation, other.innovation) {
            if left != right {
                return Err(GeneError::InnovationMismatch { left, right });
            }
        }

        let weight = choose_f64(self.weight, other.weight, rng);
        let mut enabled = choose_bool(self.enabled, other.enabled, rng);
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
                left: format!("{:?}", self.key),
                right: format!("{:?}", other.key),
            });
        }

        let mut distance = (self.weight - other.weight).abs();
        if self.enabled != other.enabled {
            distance += config.compatibility_enable_penalty;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }
}

fn choose_first(rng: &mut impl RandomSource) -> bool {
    rng.next_f64() > 0.5
}

fn choose_f64(first: f64, second: f64, rng: &mut impl RandomSource) -> f64 {
    if choose_first(rng) {
        first
    } else {
        second
    }
}

fn choose_bool(first: bool, second: bool, rng: &mut impl RandomSource) -> bool {
    if choose_first(rng) {
        first
    } else {
        second
    }
}

fn choose_string(first: &str, second: &str, rng: &mut impl RandomSource) -> String {
    if choose_first(rng) {
        first.to_string()
    } else {
        second.to_string()
    }
}
