use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::activation::{ActivationError, ActivationFunction};
use crate::aggregation::{AggregationError, AggregationFunction};
use crate::config::{GenomeConfig, NodeMemoryKind};
use crate::gene::{DefaultNodeGene, NodeKey};
use crate::genome::{input_keys, output_keys, DefaultGenome};
use crate::graph::required_for_output;
use crate::network_impl::recurrent_memory::{
    eval_node_memory, NodeGruMemory, NodeHebbianMemory, RecurrentMemoryState, RecurrentNodeMemory,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentNodeEval {
    pub node: NodeKey,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub bias: f64,
    pub response: f64,
    pub links: Vec<RecurrentConnectionEval>,
    pub memory: RecurrentNodeMemory,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentConnectionEval {
    pub input: NodeKey,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentNetwork {
    pub input_nodes: Vec<NodeKey>,
    pub output_nodes: Vec<NodeKey>,
    pub node_evals: Vec<RecurrentNodeEval>,
    dense_node_evals: Vec<DenseRecurrentNodeEval>,
    output_sources: Vec<RecurrentValueSource>,
    input_values: Vec<f64>,
    values: [Vec<f64>; 2],
    memory_states: Vec<RecurrentMemoryState>,
    active: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct DenseRecurrentNodeEval {
    node: NodeKey,
    activation: ActivationFunction,
    aggregation: AggregationFunction,
    bias: f64,
    response: f64,
    links: Vec<DenseRecurrentConnectionEval>,
    memory: RecurrentNodeMemory,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DenseRecurrentConnectionEval {
    source: RecurrentValueSource,
    weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecurrentValueSource {
    Input(usize),
    Node(usize),
    Zero,
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
        let input_index = input_nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect::<BTreeMap<_, _>>();
        let node_index = node_evals
            .iter()
            .enumerate()
            .map(|(index, node_eval)| (node_eval.node, index))
            .collect::<BTreeMap<_, _>>();
        let dense_node_evals = node_evals
            .iter()
            .map(|node_eval| DenseRecurrentNodeEval {
                node: node_eval.node,
                activation: node_eval.activation,
                aggregation: node_eval.aggregation,
                bias: node_eval.bias,
                response: node_eval.response,
                links: node_eval
                    .links
                    .iter()
                    .map(|link| DenseRecurrentConnectionEval {
                        source: resolve_source(link.input, &input_index, &node_index),
                        weight: link.weight,
                    })
                    .collect(),
                memory: node_eval.memory,
            })
            .collect::<Vec<_>>();
        let output_sources = output_nodes
            .iter()
            .copied()
            .map(|key| resolve_source(key, &input_index, &node_index))
            .collect();
        let node_count = dense_node_evals.len();

        Self {
            input_nodes,
            output_nodes,
            node_evals,
            dense_node_evals,
            output_sources,
            input_values: vec![0.0; input_index.len()],
            values: [vec![0.0; node_count], vec![0.0; node_count]],
            memory_states: vec![RecurrentMemoryState::default(); node_count],
            active: 0,
        }
    }

    pub fn reset(&mut self) {
        self.input_values.fill(0.0);
        for value_map in &mut self.values {
            value_map.fill(0.0);
        }
        self.memory_states.fill(RecurrentMemoryState::default());
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
        self.input_values.copy_from_slice(inputs);

        for node_index in 0..self.dense_node_evals.len() {
            let node_eval = &self.dense_node_evals[node_index];
            let aggregated = node_eval.aggregation.apply_iter(
                node_eval
                    .links
                    .iter()
                    .map(|link| self.source_value(link.source, input_index) * link.weight),
            );
            let previous = self.values[input_index][node_index];
            let candidate_pre = node_eval.bias + node_eval.response * aggregated;
            let update = eval_node_memory(
                node_eval.memory,
                |value| node_eval.activation.apply(value),
                candidate_pre,
                aggregated,
                previous,
                self.memory_states[node_index],
            );
            self.memory_states[node_index] = update.state;
            self.values[output_index][node_index] = update.output;
        }

        Ok(self
            .output_sources
            .iter()
            .map(|source| self.source_value(*source, output_index))
            .collect())
    }

    pub fn create(genome: &DefaultGenome, config: &GenomeConfig) -> Result<Self, RecurrentError> {
        let config_input_keys = input_keys(config);
        let config_output_keys = output_keys(config);
        let all_connections = genome.connections.keys().copied().collect::<Vec<_>>();
        let required =
            required_for_output(&config_input_keys, &config_output_keys, &all_connections);
        let mut node_inputs: BTreeMap<NodeKey, Vec<RecurrentConnectionEval>> = BTreeMap::new();

        for connection_gene in genome.connections.values() {
            if !connection_gene.enabled {
                continue;
            }

            let input_node = connection_gene.key.input;
            let output_node = connection_gene.key.output;
            if !required.contains(&output_node) && !required.contains(&input_node) {
                continue;
            }

            node_inputs
                .entry(output_node)
                .or_default()
                .push(RecurrentConnectionEval {
                    input: input_node,
                    weight: connection_gene.weight,
                });
        }

        let mut node_evals = Vec::new();
        for (node_key, links) in node_inputs {
            let node_gene = genome
                .nodes
                .get(&node_key)
                .ok_or(RecurrentError::MissingNodeGene(node_key))?;
            node_evals.push(RecurrentNodeEval::from_gene(node_key, node_gene, links));
        }

        Ok(Self::new(config_input_keys, config_output_keys, node_evals))
    }

    fn source_value(&self, source: RecurrentValueSource, value_index: usize) -> f64 {
        match source {
            RecurrentValueSource::Input(index) => self.input_values[index],
            RecurrentValueSource::Node(index) => self.values[value_index][index],
            RecurrentValueSource::Zero => 0.0,
        }
    }
}

impl RecurrentNodeEval {
    pub fn from_gene(
        node: NodeKey,
        gene: &DefaultNodeGene,
        links: Vec<RecurrentConnectionEval>,
    ) -> Self {
        Self {
            node,
            activation: gene.activation,
            aggregation: gene.aggregation,
            bias: gene.bias,
            response: gene.response,
            links,
            memory: memory_from_gene(gene),
        }
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

fn resolve_source(
    key: NodeKey,
    input_index: &BTreeMap<NodeKey, usize>,
    node_index: &BTreeMap<NodeKey, usize>,
) -> RecurrentValueSource {
    if let Some(index) = input_index.get(&key).copied() {
        RecurrentValueSource::Input(index)
    } else if let Some(index) = node_index.get(&key).copied() {
        RecurrentValueSource::Node(index)
    } else {
        RecurrentValueSource::Zero
    }
}

fn memory_from_gene(gene: &DefaultNodeGene) -> RecurrentNodeMemory {
    match gene.node_memory_kind {
        NodeMemoryKind::None => RecurrentNodeMemory::None,
        NodeMemoryKind::NodeGru => RecurrentNodeMemory::NodeGru(NodeGruMemory {
            topology: gene.node_gru_topology,
            reset_bias: gene.node_gru_reset_bias,
            reset_response: gene.node_gru_reset_response,
            reset_memory_weight: gene.node_gru_reset_memory_weight,
            update_bias: gene.node_gru_update_bias,
            update_response: gene.node_gru_update_response,
            update_memory_weight: gene.node_gru_update_memory_weight,
            candidate_memory_weight: gene.node_gru_candidate_memory_weight,
        }),
        NodeMemoryKind::Hebbian => RecurrentNodeMemory::Hebbian(NodeHebbianMemory {
            rule: gene.node_hebbian_rule,
            decay: gene.node_hebbian_decay,
            eta: gene.node_hebbian_eta,
            key_weight: gene.node_hebbian_key_weight,
            alpha: gene.node_hebbian_alpha,
            mod_bias: gene.node_hebbian_mod_bias,
            mod_response: gene.node_hebbian_mod_response,
            theta_decay: gene.node_hebbian_theta_decay,
        }),
    }
}
