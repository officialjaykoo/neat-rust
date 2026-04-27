use crate::network_impl::recurrent_memory::{
    NodeGruMemory, NodeHebbianMemory, RecurrentNodeMemory,
};

use super::{CompiledPolicyNodeEval, CompiledPolicyNodeMemory};

pub(super) fn policy_node_memory(node: &CompiledPolicyNodeEval) -> RecurrentNodeMemory {
    match &node.memory {
        CompiledPolicyNodeMemory::None => RecurrentNodeMemory::None,
        CompiledPolicyNodeMemory::NodeGru {
            topology,
            reset_bias,
            reset_response,
            reset_memory_weight,
            update_bias,
            update_response,
            update_memory_weight,
            candidate_memory_weight,
        } => RecurrentNodeMemory::NodeGru(NodeGruMemory {
            topology: topology.0,
            reset_bias: *reset_bias,
            reset_response: *reset_response,
            reset_memory_weight: *reset_memory_weight,
            update_bias: *update_bias,
            update_response: *update_response,
            update_memory_weight: *update_memory_weight,
            candidate_memory_weight: *candidate_memory_weight,
        }),
        CompiledPolicyNodeMemory::Hebbian {
            rule,
            decay,
            eta,
            key_weight,
            alpha,
            mod_bias,
            mod_response,
            theta_decay,
        } => RecurrentNodeMemory::Hebbian(NodeHebbianMemory {
            rule: rule.0,
            decay: *decay,
            eta: *eta,
            key_weight: *key_weight,
            alpha: *alpha,
            mod_bias: *mod_bias,
            mod_response: *mod_response,
            theta_decay: *theta_decay,
        }),
    }
}
