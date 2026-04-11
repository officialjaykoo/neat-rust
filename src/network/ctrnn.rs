use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::activation::{ActivationError, ActivationFunction};
use crate::aggregation::{AggregationError, AggregationFunction};
use crate::config::GenomeConfig;
use crate::gene::NodeKey;
use crate::genome::{input_keys, output_keys, DefaultGenome};
use crate::graph::required_for_output;

#[derive(Debug, Clone, PartialEq)]
pub struct CtrnnNodeEval {
    pub time_constant: f64,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub bias: f64,
    pub response: f64,
    pub links: Vec<(NodeKey, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ctrnn {
    pub input_nodes: Vec<NodeKey>,
    pub output_nodes: Vec<NodeKey>,
    pub node_evals: BTreeMap<NodeKey, CtrnnNodeEval>,
    values: [BTreeMap<NodeKey, f64>; 2],
    active: usize,
    time_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CtrnnError {
    InputCountMismatch { expected: usize, actual: usize },
    MissingNodeGene(NodeKey),
    UnknownActivation(ActivationError),
    UnknownAggregation(AggregationError),
    MaxTimeStepUnavailable,
    InvalidTimeStep,
}

impl Ctrnn {
    pub fn new(
        input_nodes: Vec<NodeKey>,
        output_nodes: Vec<NodeKey>,
        node_evals: BTreeMap<NodeKey, CtrnnNodeEval>,
    ) -> Self {
        let mut values = [BTreeMap::new(), BTreeMap::new()];
        for value_map in &mut values {
            for key in input_nodes.iter().chain(output_nodes.iter()) {
                value_map.insert(*key, 0.0);
            }
            for (node, node_eval) in &node_evals {
                value_map.insert(*node, 0.0);
                for (input_node, _) in &node_eval.links {
                    value_map.insert(*input_node, 0.0);
                }
            }
        }

        Self {
            input_nodes,
            output_nodes,
            node_evals,
            values,
            active: 0,
            time_seconds: 0.0,
        }
    }

    pub fn reset(&mut self) {
        for value_map in &mut self.values {
            for value in value_map.values_mut() {
                *value = 0.0;
            }
        }
        self.active = 0;
        self.time_seconds = 0.0;
    }

    pub fn set_node_value(&mut self, node_key: NodeKey, value: f64) {
        for value_map in &mut self.values {
            value_map.insert(node_key, value);
        }
    }

    pub fn time_seconds(&self) -> f64 {
        self.time_seconds
    }

    pub fn get_max_time_step(&self) -> Result<f64, CtrnnError> {
        Err(CtrnnError::MaxTimeStepUnavailable)
    }

    pub fn advance(
        &mut self,
        inputs: &[f64],
        advance_time: f64,
        time_step: Option<f64>,
    ) -> Result<Vec<f64>, CtrnnError> {
        if self.input_nodes.len() != inputs.len() {
            return Err(CtrnnError::InputCountMismatch {
                expected: self.input_nodes.len(),
                actual: inputs.len(),
            });
        }

        let final_time_seconds = self.time_seconds + advance_time;
        let time_step = match time_step {
            Some(value) => value,
            None if self.time_seconds < final_time_seconds => self.get_max_time_step()? * 0.5,
            None => 0.0,
        };

        if self.time_seconds < final_time_seconds && (!time_step.is_finite() || time_step <= 0.0) {
            return Err(CtrnnError::InvalidTimeStep);
        }

        while self.time_seconds < final_time_seconds {
            let dt = time_step.min(final_time_seconds - self.time_seconds);
            let input_index = self.active;
            let output_index = 1 - self.active;
            self.active = output_index;

            for (key, value) in self.input_nodes.iter().zip(inputs.iter()) {
                self.values[input_index].insert(*key, *value);
                self.values[output_index].insert(*key, *value);
            }

            for (node_key, node_eval) in &self.node_evals {
                let mut node_inputs = Vec::with_capacity(node_eval.links.len());
                for (input_node, weight) in &node_eval.links {
                    let value = self.values[input_index]
                        .get(input_node)
                        .copied()
                        .unwrap_or(0.0);
                    node_inputs.push(value * *weight);
                }

                let aggregated = node_eval.aggregation.apply(&node_inputs);
                let target = node_eval
                    .activation
                    .apply(node_eval.bias + node_eval.response * aggregated);
                let current = self.values[input_index]
                    .get(node_key)
                    .copied()
                    .unwrap_or(0.0);
                let decay = (-dt / node_eval.time_constant).exp();
                let next = decay * current + (1.0 - decay) * target;
                self.values[output_index].insert(*node_key, next);
            }

            self.time_seconds += dt;
        }

        let output_values = &self.values[self.active];
        Ok(self
            .output_nodes
            .iter()
            .map(|key| output_values.get(key).copied().unwrap_or(0.0))
            .collect())
    }

    pub fn create(genome: &DefaultGenome, config: &GenomeConfig) -> Result<Self, CtrnnError> {
        let config_input_keys = input_keys(config);
        let config_output_keys = output_keys(config);
        let all_connections: Vec<_> = genome.connections.keys().copied().collect();
        let required =
            required_for_output(&config_input_keys, &config_output_keys, &all_connections);
        let mut node_inputs: BTreeMap<NodeKey, Vec<(NodeKey, f64)>> = BTreeMap::new();

        for connection_gene in genome.connections.values() {
            if !connection_gene.enabled {
                continue;
            }

            let (input_node, output_node) = connection_gene.key;
            if !required.contains(&output_node) && !required.contains(&input_node) {
                continue;
            }

            node_inputs
                .entry(output_node)
                .or_default()
                .push((input_node, connection_gene.weight));
        }

        let mut node_evals = BTreeMap::new();
        for (node_key, links) in node_inputs {
            let node_gene = genome
                .nodes
                .get(&node_key)
                .ok_or(CtrnnError::MissingNodeGene(node_key))?;
            let aggregation =
                AggregationFunction::from_name(&node_gene.aggregation).ok_or_else(|| {
                    CtrnnError::UnknownAggregation(AggregationError::unknown(
                        &node_gene.aggregation,
                    ))
                })?;
            let activation =
                ActivationFunction::from_name(&node_gene.activation).ok_or_else(|| {
                    CtrnnError::UnknownActivation(ActivationError::unknown(&node_gene.activation))
                })?;

            node_evals.insert(
                node_key,
                CtrnnNodeEval {
                    time_constant: node_gene.time_constant,
                    activation,
                    aggregation,
                    bias: node_gene.bias,
                    response: node_gene.response,
                    links,
                },
            );
        }

        Ok(Self::new(config_input_keys, config_output_keys, node_evals))
    }
}

impl fmt::Display for CtrnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::MissingNodeGene(node) => write!(f, "missing node gene for node {node}"),
            Self::UnknownActivation(err) => write!(f, "{err}"),
            Self::UnknownAggregation(err) => write!(f, "{err}"),
            Self::MaxTimeStepUnavailable => write!(f, "max CTRNN time step is not implemented"),
            Self::InvalidTimeStep => write!(f, "CTRNN time_step must be finite and greater than 0"),
        }
    }
}

impl Error for CtrnnError {}
