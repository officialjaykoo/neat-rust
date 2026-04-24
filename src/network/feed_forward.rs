use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::activation::{ActivationError, ActivationFunction};
use crate::aggregation::{AggregationError, AggregationFunction};
use crate::config::GenomeConfig;
use crate::gene::{ConnectionKey, NodeKey};
use crate::genome::{input_keys, output_keys, DefaultGenome};
use crate::graph::feed_forward_layers;

#[derive(Debug, Clone, PartialEq)]
pub struct NodeEval {
    pub node: NodeKey,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub bias: f64,
    pub response: f64,
    pub links: Vec<(NodeKey, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork {
    pub input_nodes: Vec<NodeKey>,
    pub output_nodes: Vec<NodeKey>,
    pub node_evals: Vec<NodeEval>,
    values: BTreeMap<NodeKey, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedForwardError {
    InputCountMismatch { expected: usize, actual: usize },
    MissingNodeGene(NodeKey),
    MissingNodeValue(NodeKey),
    UnknownActivation(ActivationError),
    UnknownAggregation(AggregationError),
}

impl FeedForwardNetwork {
    pub fn new(
        input_nodes: Vec<NodeKey>,
        output_nodes: Vec<NodeKey>,
        node_evals: Vec<NodeEval>,
    ) -> Self {
        let mut values = BTreeMap::new();
        for key in input_nodes.iter().chain(output_nodes.iter()) {
            values.insert(*key, 0.0);
        }

        Self {
            input_nodes,
            output_nodes,
            node_evals,
            values,
        }
    }

    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, FeedForwardError> {
        if self.input_nodes.len() != inputs.len() {
            return Err(FeedForwardError::InputCountMismatch {
                expected: self.input_nodes.len(),
                actual: inputs.len(),
            });
        }

        for (key, value) in self.input_nodes.iter().zip(inputs.iter()) {
            self.values.insert(*key, *value);
        }

        for node_eval in &self.node_evals {
            let mut node_inputs = Vec::with_capacity(node_eval.links.len());
            for (input_node, weight) in &node_eval.links {
                let value = self
                    .values
                    .get(input_node)
                    .copied()
                    .ok_or(FeedForwardError::MissingNodeValue(*input_node))?;
                node_inputs.push(value * *weight);
            }
            let aggregated = node_eval.aggregation.apply(&node_inputs);
            let pre_activation = node_eval.bias + node_eval.response * aggregated;
            self.values
                .insert(node_eval.node, node_eval.activation.apply(pre_activation));
        }

        Ok(self
            .output_nodes
            .iter()
            .map(|key| self.values.get(key).copied().unwrap_or(0.0))
            .collect())
    }

    pub fn create(genome: &DefaultGenome, config: &GenomeConfig) -> Result<Self, FeedForwardError> {
        let expressed_connections: Vec<ConnectionKey> = genome
            .connections
            .values()
            .filter(|connection| connection.enabled)
            .map(|connection| connection.key)
            .collect();
        let config_input_keys = input_keys(config);
        let config_output_keys = output_keys(config);
        let (layers, required) = feed_forward_layers(
            &config_input_keys,
            &config_output_keys,
            &expressed_connections,
        );
        let mut required_with_inputs: BTreeSet<NodeKey> = required;
        required_with_inputs.extend(config_input_keys.iter().copied());

        let mut node_evals = Vec::new();
        for layer in layers {
            for node in layer {
                let mut links = Vec::new();
                for connection_key in &expressed_connections {
                    let input_node = connection_key.input;
                    let output_node = connection_key.output;
                    if output_node == node && required_with_inputs.contains(&input_node) {
                        if let Some(connection_gene) = genome.connections.get(connection_key) {
                            links.push((input_node, connection_gene.weight));
                        }
                    }
                }

                let node_gene = genome
                    .nodes
                    .get(&node)
                    .ok_or(FeedForwardError::MissingNodeGene(node))?;
                node_evals.push(NodeEval {
                    node,
                    activation: node_gene.activation,
                    aggregation: node_gene.aggregation,
                    bias: node_gene.bias,
                    response: node_gene.response,
                    links,
                });
            }
        }

        Ok(Self::new(config_input_keys, config_output_keys, node_evals))
    }
}

impl fmt::Display for FeedForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::MissingNodeGene(node) => write!(f, "missing node gene for node {node}"),
            Self::MissingNodeValue(node) => write!(f, "missing node value for node {node}"),
            Self::UnknownActivation(err) => write!(f, "{err}"),
            Self::UnknownAggregation(err) => write!(f, "{err}"),
        }
    }
}

impl Error for FeedForwardError {}
