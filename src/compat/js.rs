//! JS-facing bridge surface.

pub use crate::eval_bridge::{
    default_node_bin, run_neat_eval_worker, BridgeEarlyStopConfig, BridgeGameCount,
    BridgeJsonArrayArg, BridgeNativeInferenceBackend, BridgeOpponent, BridgeSeat, BridgeStepCount,
    BridgeTurnPolicy, EvalBridgeError, EvalBridgeOptions, EvalBridgeOutput, EvalSeed, NodeCommand,
};
pub use crate::policy_bridge::{
    evaluate_policy_batch, native_policy_cuda_available, CompiledPolicyNodeEval,
    CompiledPolicyRequest, CompiledPolicyResult, CompiledPolicySnapshot, CompiledPolicySpec,
    PolicyBridgeBackend, PolicyBridgeError, PolicyIncomingEdge,
};
