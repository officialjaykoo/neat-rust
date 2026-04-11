use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::config::GenomeConfig;
use crate::gene::NodeKey;
use crate::genome::{input_keys, output_keys, DefaultGenome};
use crate::graph::required_for_output;

pub const REGULAR_SPIKING_PARAMS: IzParams = IzParams {
    a: 0.02,
    b: 0.20,
    c: -65.0,
    d: 8.0,
};
pub const INTRINSICALLY_BURSTING_PARAMS: IzParams = IzParams {
    a: 0.02,
    b: 0.20,
    c: -55.0,
    d: 4.0,
};
pub const CHATTERING_PARAMS: IzParams = IzParams {
    a: 0.02,
    b: 0.20,
    c: -50.0,
    d: 2.0,
};
pub const FAST_SPIKING_PARAMS: IzParams = IzParams {
    a: 0.10,
    b: 0.20,
    c: -65.0,
    d: 2.0,
};
pub const THALAMO_CORTICAL_PARAMS: IzParams = IzParams {
    a: 0.02,
    b: 0.25,
    c: -65.0,
    d: 0.05,
};
pub const RESONATOR_PARAMS: IzParams = IzParams {
    a: 0.10,
    b: 0.25,
    c: -65.0,
    d: 2.0,
};
pub const LOW_THRESHOLD_SPIKING_PARAMS: IzParams = IzParams {
    a: 0.02,
    b: 0.25,
    c: -65.0,
    d: 2.0,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IzParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IzNeuron {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub bias: f64,
    pub inputs: Vec<(NodeKey, f64)>,
    pub v: f64,
    pub u: f64,
    pub fired: f64,
    pub current: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Iznn {
    pub neurons: BTreeMap<NodeKey, IzNeuron>,
    pub inputs: Vec<NodeKey>,
    pub outputs: Vec<NodeKey>,
    input_values: BTreeMap<NodeKey, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IznnError {
    InputCountMismatch { expected: usize, actual: usize },
    MissingInputValue(NodeKey),
    MissingNodeGene(NodeKey),
}

impl IzNeuron {
    pub fn new(bias: f64, params: IzParams, inputs: Vec<(NodeKey, f64)>) -> Self {
        let v = params.c;
        let u = params.b * v;
        Self {
            a: params.a,
            b: params.b,
            c: params.c,
            d: params.d,
            bias,
            inputs,
            v,
            u,
            fired: 0.0,
            current: bias,
        }
    }

    pub fn advance(&mut self, dt_msec: f64) {
        self.v +=
            0.5 * dt_msec * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + self.current);
        self.v +=
            0.5 * dt_msec * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + self.current);
        self.u += dt_msec * self.a * (self.b * self.v - self.u);

        self.fired = 0.0;
        if !self.v.is_finite() || !self.u.is_finite() {
            self.v = self.c;
            self.u = self.b * self.v;
            return;
        }

        if self.v > 30.0 {
            self.fired = 1.0;
            self.v = self.c;
            self.u += self.d;
        }
    }

    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.v;
        self.fired = 0.0;
        self.current = self.bias;
    }
}

impl Iznn {
    pub fn new(
        neurons: BTreeMap<NodeKey, IzNeuron>,
        inputs: Vec<NodeKey>,
        outputs: Vec<NodeKey>,
    ) -> Self {
        Self {
            neurons,
            inputs,
            outputs,
            input_values: BTreeMap::new(),
        }
    }

    pub fn set_inputs(&mut self, inputs: &[f64]) -> Result<(), IznnError> {
        if self.inputs.len() != inputs.len() {
            return Err(IznnError::InputCountMismatch {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        for (key, value) in self.inputs.iter().zip(inputs.iter()) {
            self.input_values.insert(*key, *value);
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for neuron in self.neurons.values_mut() {
            neuron.reset();
        }
    }

    pub fn get_time_step_msec(&self) -> f64 {
        0.05
    }

    pub fn advance(&mut self, dt_msec: f64) -> Result<Vec<f64>, IznnError> {
        let fired_snapshot: BTreeMap<NodeKey, f64> = self
            .neurons
            .iter()
            .map(|(key, neuron)| (*key, neuron.fired))
            .collect();
        let mut currents = BTreeMap::new();

        for (key, neuron) in &self.neurons {
            let mut current = neuron.bias;
            for (input_node, weight) in &neuron.inputs {
                let input_value = if let Some(fired) = fired_snapshot.get(input_node) {
                    *fired
                } else {
                    *self
                        .input_values
                        .get(input_node)
                        .ok_or(IznnError::MissingInputValue(*input_node))?
                };
                current += input_value * *weight;
            }
            currents.insert(*key, current);
        }

        for (key, current) in currents {
            if let Some(neuron) = self.neurons.get_mut(&key) {
                neuron.current = current;
                neuron.advance(dt_msec);
            }
        }

        Ok(self
            .outputs
            .iter()
            .map(|key| {
                self.neurons
                    .get(key)
                    .map(|neuron| neuron.fired)
                    .unwrap_or(0.0)
            })
            .collect())
    }

    pub fn create(genome: &DefaultGenome, config: &GenomeConfig) -> Result<Self, IznnError> {
        let config_input_keys = input_keys(config);
        let config_output_keys = output_keys(config);
        let input_set: BTreeSet<NodeKey> = config_input_keys.iter().copied().collect();
        let all_connections: Vec<_> = genome.connections.keys().copied().collect();
        let required =
            required_for_output(&config_input_keys, &config_output_keys, &all_connections);
        let mut node_inputs: BTreeMap<NodeKey, Vec<(NodeKey, f64)>> = BTreeMap::new();

        for connection_gene in genome.connections.values() {
            if !connection_gene.enabled {
                continue;
            }

            let (input_node, output_node) = connection_gene.key;
            if !required.contains(&output_node) {
                continue;
            }
            if !required.contains(&input_node) && !input_set.contains(&input_node) {
                continue;
            }

            node_inputs
                .entry(output_node)
                .or_default()
                .push((input_node, connection_gene.weight));
        }

        let mut neurons = BTreeMap::new();
        for node_key in required {
            let node_gene = genome
                .nodes
                .get(&node_key)
                .ok_or(IznnError::MissingNodeGene(node_key))?;
            let params = IzParams {
                a: node_gene.iz_a,
                b: node_gene.iz_b,
                c: node_gene.iz_c,
                d: node_gene.iz_d,
            };
            neurons.insert(
                node_key,
                IzNeuron::new(
                    node_gene.bias,
                    params,
                    node_inputs.remove(&node_key).unwrap_or_default(),
                ),
            );
        }

        Ok(Self::new(neurons, config_input_keys, config_output_keys))
    }
}

impl fmt::Display for IznnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::MissingInputValue(node) => write!(f, "missing input value for node {node}"),
            Self::MissingNodeGene(node) => write!(f, "missing node gene for node {node}"),
        }
    }
}

impl Error for IznnError {}
