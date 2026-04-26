use crate::config::{NodeGruTopology, NodeHebbianRule};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecurrentNodeMemory {
    None,
    NodeGru(NodeGruMemory),
    Hebbian(NodeHebbianMemory),
    LinearGate(NodeLinearGateMemory),
    LinearGateV2(NodeLinearGateV2Memory),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeGruMemory {
    pub topology: NodeGruTopology,
    pub reset_bias: f64,
    pub reset_response: f64,
    pub reset_memory_weight: f64,
    pub update_bias: f64,
    pub update_response: f64,
    pub update_memory_weight: f64,
    pub candidate_memory_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeHebbianMemory {
    pub rule: NodeHebbianRule,
    pub decay: f64,
    pub eta: f64,
    pub key_weight: f64,
    pub alpha: f64,
    pub mod_bias: f64,
    pub mod_response: f64,
    pub theta_decay: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLinearGateMemory {
    pub decay_bias: f64,
    pub decay_response: f64,
    pub write_weight: f64,
    pub gate_bias: f64,
    pub gate_response: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLinearGateV2Memory {
    pub decay_bias: f64,
    pub decay_response: f64,
    pub write_weight: f64,
    pub gate_bias: f64,
    pub gate_response: f64,
    pub min_decay: f64,
    pub input_mix: f64,
    pub memory_weight: f64,
    pub trace_decay: f64,
    pub trace_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentMemoryState {
    pub fast_weight: f64,
    pub threshold: f64,
    pub linear_trace: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentMemoryUpdate {
    pub output: f64,
    pub state: RecurrentMemoryState,
}

impl Default for RecurrentMemoryState {
    fn default() -> Self {
        Self {
            fast_weight: 0.0,
            threshold: 0.0,
            linear_trace: 0.0,
        }
    }
}

pub(crate) fn eval_node_memory(
    memory: RecurrentNodeMemory,
    activation: impl Fn(f64) -> f64 + Copy,
    candidate_pre: f64,
    aggregated: f64,
    previous: f64,
    state: RecurrentMemoryState,
) -> RecurrentMemoryUpdate {
    let candidate = activation(candidate_pre);
    match memory {
        RecurrentNodeMemory::None => RecurrentMemoryUpdate {
            output: candidate,
            state,
        },
        RecurrentNodeMemory::NodeGru(memory) => eval_node_gru(
            memory,
            activation,
            candidate_pre,
            aggregated,
            previous,
            state,
        ),
        RecurrentNodeMemory::Hebbian(memory) => {
            eval_hebbian(memory, activation, candidate_pre, aggregated, state)
        }
        RecurrentNodeMemory::LinearGate(memory) => {
            eval_linear_gate(memory, candidate, aggregated, previous, state)
        }
        RecurrentNodeMemory::LinearGateV2(memory) => {
            eval_linear_gate_v2(memory, candidate, aggregated, previous, state)
        }
    }
}

pub(crate) fn sigmoid_gate(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn eval_node_gru(
    memory: NodeGruMemory,
    activation: impl Fn(f64) -> f64 + Copy,
    candidate_pre: f64,
    aggregated: f64,
    previous: f64,
    state: RecurrentMemoryState,
) -> RecurrentMemoryUpdate {
    let update_gate = match memory.topology {
        NodeGruTopology::ResetOnly => 0.0,
        _ => sigmoid_gate(
            memory.update_bias
                + memory.update_response * aggregated
                + memory.update_memory_weight * previous,
        ),
    };
    let reset_gate = match memory.topology {
        NodeGruTopology::Minimal => 1.0,
        NodeGruTopology::Coupled => 1.0 - update_gate,
        NodeGruTopology::Standard | NodeGruTopology::ResetOnly => sigmoid_gate(
            memory.reset_bias
                + memory.reset_response * aggregated
                + memory.reset_memory_weight * previous,
        ),
    };
    let gru_candidate =
        activation(candidate_pre + reset_gate * previous * memory.candidate_memory_weight);
    RecurrentMemoryUpdate {
        output: update_gate * previous + (1.0 - update_gate) * gru_candidate,
        state,
    }
}

fn eval_hebbian(
    memory: NodeHebbianMemory,
    activation: impl Fn(f64) -> f64 + Copy,
    candidate_pre: f64,
    aggregated: f64,
    mut state: RecurrentMemoryState,
) -> RecurrentMemoryUpdate {
    let key = aggregated * memory.key_weight;
    let output = activation(candidate_pre + memory.alpha * state.fast_weight * key);
    let output_sq = output * output;
    let theta_decay = memory.theta_decay.clamp(0.0, 0.999);
    let next_threshold = theta_decay * state.threshold + (1.0 - theta_decay) * output_sq;
    let plain = key * output;
    let oja = plain - output_sq * state.fast_weight;
    let bcm = plain * (output - state.threshold);
    let plastic = match memory.rule {
        NodeHebbianRule::Plain => plain,
        NodeHebbianRule::Oja => oja,
        NodeHebbianRule::Bcm => bcm,
        NodeHebbianRule::OjaBcm => 0.5 * (oja + bcm),
    };
    let modulation = sigmoid_gate(memory.mod_bias + memory.mod_response * aggregated);
    state.fast_weight = (memory.decay.clamp(0.0, 1.0) * state.fast_weight
        + modulation * memory.eta * plastic)
        .clamp(-2.0, 2.0);
    state.threshold = next_threshold.clamp(0.0, 4.0);
    RecurrentMemoryUpdate { output, state }
}

fn eval_linear_gate(
    memory: NodeLinearGateMemory,
    candidate: f64,
    aggregated: f64,
    previous: f64,
    state: RecurrentMemoryState,
) -> RecurrentMemoryUpdate {
    let decay = sigmoid_gate(memory.decay_bias + memory.decay_response * aggregated);
    let write = candidate * memory.write_weight;
    let recurrent_state = (decay * previous + (1.0 - decay) * write).clamp(-1.0, 1.0);
    let gate = sigmoid_gate(memory.gate_bias + memory.gate_response * aggregated);
    RecurrentMemoryUpdate {
        output: gate * recurrent_state + (1.0 - gate) * candidate,
        state,
    }
}

fn eval_linear_gate_v2(
    memory: NodeLinearGateV2Memory,
    candidate: f64,
    aggregated: f64,
    previous: f64,
    mut state: RecurrentMemoryState,
) -> RecurrentMemoryUpdate {
    let mixed = aggregated + memory.input_mix * previous;
    let raw_decay = sigmoid_gate(memory.decay_bias + memory.decay_response * mixed);
    let min_decay = memory.min_decay.clamp(0.0, 0.99);
    let decay = min_decay + (1.0 - min_decay) * raw_decay;
    let write_gate = sigmoid_gate(memory.gate_bias + memory.gate_response * mixed);
    let trace_decay = memory.trace_decay.clamp(0.0, 0.99);
    let next_trace =
        trace_decay * state.linear_trace + (1.0 - trace_decay) * (mixed * candidate).tanh();
    let write = (candidate * memory.write_weight + memory.trace_weight * state.linear_trace).tanh();
    let recurrent_state = (decay * previous + (1.0 - decay) * write_gate * write).clamp(-1.0, 1.0);
    let output = (candidate
        + memory.memory_weight.clamp(0.0, 1.5) * (recurrent_state - candidate)
        + memory.trace_weight * next_trace)
        .clamp(-1.0, 1.0);
    state.linear_trace = next_trace.clamp(-1.0, 1.0);
    RecurrentMemoryUpdate { output, state }
}
