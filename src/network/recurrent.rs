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
pub struct RecurrentNodeEval {
    pub node: NodeKey,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub bias: f64,
    pub response: f64,
    pub links: Vec<(NodeKey, f64)>,
    pub memory_gate_enabled: bool,
    pub memory_gate_bias: f64,
    pub memory_gate_response: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentNetwork {
    pub input_nodes: Vec<NodeKey>,
    pub output_nodes: Vec<NodeKey>,
    pub node_evals: Vec<RecurrentNodeEval>,
    values: [BTreeMap<NodeKey, f64>; 2],
    active: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecurrentError {
    InputCountMismatch { expected: usize, actual: usize },
    MissingNodeGene(NodeKey),
    UnknownActivation(ActivationError),
    UnknownAggregation(AggregationError),
}

impl RecurrentNetwork {
    pub fn new(
        input_nodes: Vec<NodeKey>,
        output_nodes: Vec<NodeKey>,
        node_evals: Vec<RecurrentNodeEval>,
    ) -> Self {
        let mut values = [BTreeMap::new(), BTreeMap::new()];
        for value_map in &mut values {
            for key in input_nodes.iter().chain(output_nodes.iter()) {
                value_map.insert(*key, 0.0);
            }
            for node_eval in &node_evals {
                value_map.insert(node_eval.node, 0.0);
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
        }
    }

    pub fn reset(&mut self) {
        for value_map in &mut self.values {
            for value in value_map.values_mut() {
                *value = 0.0;
            }
        }
        self.active = 0;
    }

    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, RecurrentError> {
        if self.input_nodes.len() != inputs.len() {
            return Err(RecurrentError::InputCountMismatch {
                expected: self.input_nodes.len(),
                actual: inputs.len(),
            });
        }

        let input_index = self.active;
        let output_index = 1 - self.active;
        self.active = output_index;

        for (key, value) in self.input_nodes.iter().zip(inputs.iter()) {
            self.values[input_index].insert(*key, *value);
            self.values[output_index].insert(*key, *value);
        }

        for node_eval in &self.node_evals {
            let mut node_inputs = Vec::with_capacity(node_eval.links.len());
            for (input_node, weight) in &node_eval.links {
                let value = self.values[input_index]
                    .get(input_node)
                    .copied()
                    .unwrap_or(0.0);
                node_inputs.push(value * *weight);
            }

            let aggregated = node_eval.aggregation.apply(&node_inputs);
            let candidate = node_eval
                .activation
                .apply(node_eval.bias + node_eval.response * aggregated);
            let next_value = if node_eval.memory_gate_enabled {
                let gate = sigmoid_gate(
                    node_eval.memory_gate_bias + node_eval.memory_gate_response * aggregated,
                );
                let previous = self.values[input_index]
                    .get(&node_eval.node)
                    .copied()
                    .unwrap_or(0.0);
                (1.0 - gate) * previous + gate * candidate
            } else {
                candidate
            };
            self.values[output_index].insert(node_eval.node, next_value);
        }

        Ok(self
            .output_nodes
            .iter()
            .map(|key| self.values[output_index].get(key).copied().unwrap_or(0.0))
            .collect())
    }

    pub fn create(genome: &DefaultGenome, config: &GenomeConfig) -> Result<Self, RecurrentError> {
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

        let mut node_evals = Vec::new();
        for (node_key, links) in node_inputs {
            let node_gene = genome
                .nodes
                .get(&node_key)
                .ok_or(RecurrentError::MissingNodeGene(node_key))?;
            let aggregation =
                AggregationFunction::from_name(&node_gene.aggregation).ok_or_else(|| {
                    RecurrentError::UnknownAggregation(AggregationError::unknown(
                        &node_gene.aggregation,
                    ))
                })?;
            let activation =
                ActivationFunction::from_name(&node_gene.activation).ok_or_else(|| {
                    RecurrentError::UnknownActivation(ActivationError::unknown(
                        &node_gene.activation,
                    ))
                })?;

            node_evals.push(RecurrentNodeEval {
                node: node_key,
                activation,
                aggregation,
                bias: node_gene.bias,
                response: node_gene.response,
                links,
                memory_gate_enabled: node_gene.memory_gate_enabled,
                memory_gate_bias: node_gene.memory_gate_bias,
                memory_gate_response: node_gene.memory_gate_response,
            });
        }

        Ok(Self::new(config_input_keys, config_output_keys, node_evals))
    }
}

impl fmt::Display for RecurrentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::MissingNodeGene(node) => write!(f, "missing node gene for node {node}"),
            Self::UnknownActivation(err) => write!(f, "{err}"),
            Self::UnknownAggregation(err) => write!(f, "{err}"),
        }
    }
}

impl Error for RecurrentError {}

fn sigmoid_gate(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}
