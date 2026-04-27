use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;
use crate::attributes::{ChoiceAttribute, FloatAttribute, RandomSource};
use crate::config::{GenomeConfig, NodeGruTopology, NodeHebbianRule, NodeMemoryKind};

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
    pub node_memory_kind: NodeMemoryKind,
    pub node_gru_topology: NodeGruTopology,
    pub node_gru_reset_bias: f64,
    pub node_gru_reset_response: f64,
    pub node_gru_reset_memory_weight: f64,
    pub node_gru_update_bias: f64,
    pub node_gru_update_response: f64,
    pub node_gru_update_memory_weight: f64,
    pub node_gru_candidate_memory_weight: f64,
    pub node_hebbian_rule: NodeHebbianRule,
    pub node_hebbian_decay: f64,
    pub node_hebbian_eta: f64,
    pub node_hebbian_key_weight: f64,
    pub node_hebbian_alpha: f64,
    pub node_hebbian_mod_bias: f64,
    pub node_hebbian_mod_response: f64,
    pub node_hebbian_theta_decay: f64,
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
            node_memory_kind: NodeMemoryKind::None,
            node_gru_topology: NodeGruTopology::Standard,
            node_gru_reset_bias: 0.0,
            node_gru_reset_response: 1.0,
            node_gru_reset_memory_weight: 0.0,
            node_gru_update_bias: 0.0,
            node_gru_update_response: 1.0,
            node_gru_update_memory_weight: 0.0,
            node_gru_candidate_memory_weight: 0.0,
            node_hebbian_rule: NodeHebbianRule::Oja,
            node_hebbian_decay: 0.9,
            node_hebbian_eta: 0.05,
            node_hebbian_key_weight: 1.0,
            node_hebbian_alpha: 0.5,
            node_hebbian_mod_bias: 0.0,
            node_hebbian_mod_response: 1.0,
            node_hebbian_theta_decay: 0.95,
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
        self.node_memory_kind = ChoiceAttribute::init_value(&config.node_memory_kind, rng)?;
        self.node_gru_topology = ChoiceAttribute::init_value(&config.node_gru_topology, rng)?;
        self.node_gru_reset_bias = FloatAttribute::init_value(&config.node_gru_reset_bias, rng)?;
        self.node_gru_reset_response =
            FloatAttribute::init_value(&config.node_gru_reset_response, rng)?;
        self.node_gru_reset_memory_weight =
            FloatAttribute::init_value(&config.node_gru_reset_memory_weight, rng)?;
        self.node_gru_update_bias = FloatAttribute::init_value(&config.node_gru_update_bias, rng)?;
        self.node_gru_update_response =
            FloatAttribute::init_value(&config.node_gru_update_response, rng)?;
        self.node_gru_update_memory_weight =
            FloatAttribute::init_value(&config.node_gru_update_memory_weight, rng)?;
        self.node_gru_candidate_memory_weight =
            FloatAttribute::init_value(&config.node_gru_candidate_memory_weight, rng)?;
        self.node_hebbian_rule = ChoiceAttribute::init_value(&config.node_hebbian_rule, rng)?;
        self.node_hebbian_decay = FloatAttribute::init_value(&config.node_hebbian_decay, rng)?;
        self.node_hebbian_eta = FloatAttribute::init_value(&config.node_hebbian_eta, rng)?;
        self.node_hebbian_key_weight =
            FloatAttribute::init_value(&config.node_hebbian_key_weight, rng)?;
        self.node_hebbian_alpha = FloatAttribute::init_value(&config.node_hebbian_alpha, rng)?;
        self.node_hebbian_mod_bias =
            FloatAttribute::init_value(&config.node_hebbian_mod_bias, rng)?;
        self.node_hebbian_mod_response =
            FloatAttribute::init_value(&config.node_hebbian_mod_response, rng)?;
        self.node_hebbian_theta_decay =
            FloatAttribute::init_value(&config.node_hebbian_theta_decay, rng)?;
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
        self.node_memory_kind =
            ChoiceAttribute::mutate_value(self.node_memory_kind, &config.node_memory_kind, rng)?;
        self.node_gru_topology =
            ChoiceAttribute::mutate_value(self.node_gru_topology, &config.node_gru_topology, rng)?;
        self.node_gru_reset_bias = FloatAttribute::mutate_value(
            self.node_gru_reset_bias,
            &config.node_gru_reset_bias,
            rng,
        )?;
        self.node_gru_reset_response = FloatAttribute::mutate_value(
            self.node_gru_reset_response,
            &config.node_gru_reset_response,
            rng,
        )?;
        self.node_gru_reset_memory_weight = FloatAttribute::mutate_value(
            self.node_gru_reset_memory_weight,
            &config.node_gru_reset_memory_weight,
            rng,
        )?;
        self.node_gru_update_bias = FloatAttribute::mutate_value(
            self.node_gru_update_bias,
            &config.node_gru_update_bias,
            rng,
        )?;
        self.node_gru_update_response = FloatAttribute::mutate_value(
            self.node_gru_update_response,
            &config.node_gru_update_response,
            rng,
        )?;
        self.node_gru_update_memory_weight = FloatAttribute::mutate_value(
            self.node_gru_update_memory_weight,
            &config.node_gru_update_memory_weight,
            rng,
        )?;
        self.node_gru_candidate_memory_weight = FloatAttribute::mutate_value(
            self.node_gru_candidate_memory_weight,
            &config.node_gru_candidate_memory_weight,
            rng,
        )?;
        self.node_hebbian_rule =
            ChoiceAttribute::mutate_value(self.node_hebbian_rule, &config.node_hebbian_rule, rng)?;
        self.node_hebbian_decay =
            FloatAttribute::mutate_value(self.node_hebbian_decay, &config.node_hebbian_decay, rng)?;
        self.node_hebbian_eta =
            FloatAttribute::mutate_value(self.node_hebbian_eta, &config.node_hebbian_eta, rng)?;
        self.node_hebbian_key_weight = FloatAttribute::mutate_value(
            self.node_hebbian_key_weight,
            &config.node_hebbian_key_weight,
            rng,
        )?;
        self.node_hebbian_alpha =
            FloatAttribute::mutate_value(self.node_hebbian_alpha, &config.node_hebbian_alpha, rng)?;
        self.node_hebbian_mod_bias = FloatAttribute::mutate_value(
            self.node_hebbian_mod_bias,
            &config.node_hebbian_mod_bias,
            rng,
        )?;
        self.node_hebbian_mod_response = FloatAttribute::mutate_value(
            self.node_hebbian_mod_response,
            &config.node_hebbian_mod_response,
            rng,
        )?;
        self.node_hebbian_theta_decay = FloatAttribute::mutate_value(
            self.node_hebbian_theta_decay,
            &config.node_hebbian_theta_decay,
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
            node_memory_kind: choose_copy(self.node_memory_kind, other.node_memory_kind, rng),
            node_gru_topology: choose_copy(self.node_gru_topology, other.node_gru_topology, rng),
            node_gru_reset_bias: choose_copy(
                self.node_gru_reset_bias,
                other.node_gru_reset_bias,
                rng,
            ),
            node_gru_reset_response: choose_copy(
                self.node_gru_reset_response,
                other.node_gru_reset_response,
                rng,
            ),
            node_gru_reset_memory_weight: choose_copy(
                self.node_gru_reset_memory_weight,
                other.node_gru_reset_memory_weight,
                rng,
            ),
            node_gru_update_bias: choose_copy(
                self.node_gru_update_bias,
                other.node_gru_update_bias,
                rng,
            ),
            node_gru_update_response: choose_copy(
                self.node_gru_update_response,
                other.node_gru_update_response,
                rng,
            ),
            node_gru_update_memory_weight: choose_copy(
                self.node_gru_update_memory_weight,
                other.node_gru_update_memory_weight,
                rng,
            ),
            node_gru_candidate_memory_weight: choose_copy(
                self.node_gru_candidate_memory_weight,
                other.node_gru_candidate_memory_weight,
                rng,
            ),
            node_hebbian_rule: choose_copy(self.node_hebbian_rule, other.node_hebbian_rule, rng),
            node_hebbian_decay: choose_copy(self.node_hebbian_decay, other.node_hebbian_decay, rng),
            node_hebbian_eta: choose_copy(self.node_hebbian_eta, other.node_hebbian_eta, rng),
            node_hebbian_key_weight: choose_copy(
                self.node_hebbian_key_weight,
                other.node_hebbian_key_weight,
                rng,
            ),
            node_hebbian_alpha: choose_copy(self.node_hebbian_alpha, other.node_hebbian_alpha, rng),
            node_hebbian_mod_bias: choose_copy(
                self.node_hebbian_mod_bias,
                other.node_hebbian_mod_bias,
                rng,
            ),
            node_hebbian_mod_response: choose_copy(
                self.node_hebbian_mod_response,
                other.node_hebbian_mod_response,
                rng,
            ),
            node_hebbian_theta_decay: choose_copy(
                self.node_hebbian_theta_decay,
                other.node_hebbian_theta_decay,
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
        if self.node_memory_kind != other.node_memory_kind {
            distance += 1.0;
        } else {
            distance += self.memory_distance(other);
        }
        if self.activation != other.activation {
            distance += 1.0;
        }
        if self.aggregation != other.aggregation {
            distance += 1.0;
        }

        Ok(distance * config.compatibility_weight_coefficient)
    }

    fn memory_distance(&self, other: &Self) -> f64 {
        match self.node_memory_kind {
            NodeMemoryKind::None => 0.0,
            NodeMemoryKind::NodeGru => self.node_gru_distance(other),
            NodeMemoryKind::Hebbian => self.node_hebbian_distance(other),
        }
    }

    fn node_gru_distance(&self, other: &Self) -> f64 {
        let mut distance = 0.0;
        if self.node_gru_topology != other.node_gru_topology {
            distance += 1.0;
        }
        distance += (self.node_gru_reset_bias - other.node_gru_reset_bias).abs();
        distance += (self.node_gru_reset_response - other.node_gru_reset_response).abs();
        distance += (self.node_gru_reset_memory_weight - other.node_gru_reset_memory_weight).abs();
        distance += (self.node_gru_update_bias - other.node_gru_update_bias).abs();
        distance += (self.node_gru_update_response - other.node_gru_update_response).abs();
        distance +=
            (self.node_gru_update_memory_weight - other.node_gru_update_memory_weight).abs();
        distance
            + (self.node_gru_candidate_memory_weight - other.node_gru_candidate_memory_weight).abs()
    }

    fn node_hebbian_distance(&self, other: &Self) -> f64 {
        let mut distance = 0.0;
        if self.node_hebbian_rule != other.node_hebbian_rule {
            distance += 1.0;
        }
        distance += (self.node_hebbian_decay - other.node_hebbian_decay).abs();
        distance += (self.node_hebbian_eta - other.node_hebbian_eta).abs();
        distance += (self.node_hebbian_key_weight - other.node_hebbian_key_weight).abs();
        distance += (self.node_hebbian_alpha - other.node_hebbian_alpha).abs();
        distance += (self.node_hebbian_mod_bias - other.node_hebbian_mod_bias).abs();
        distance += (self.node_hebbian_mod_response - other.node_hebbian_mod_response).abs();
        distance + (self.node_hebbian_theta_decay - other.node_hebbian_theta_decay).abs()
    }
}
