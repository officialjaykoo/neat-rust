use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use json::JsonValue;

use crate::policy_gpu_native;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyBridgeBackend {
    Cpu,
    Auto,
    CudaNative,
}

impl PolicyBridgeBackend {
    pub fn parse(value: &str) -> Result<Self, PolicyBridgeError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" | "rust_cpu" => Ok(Self::Cpu),
            "cuda" | "cuda_native" | "native_cuda" | "rust_cuda" => Ok(Self::CudaNative),
            other => Err(PolicyBridgeError::Protocol(format!(
                "unsupported policy bridge backend '{other}'; expected one of: auto, cpu, cuda_native"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::CudaNative => "cuda_native",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyIncomingEdge {
    pub source_kind: String,
    pub source_index: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPolicyNodeEval {
    pub node_id: i64,
    pub activation: String,
    pub aggregation: String,
    pub bias: f64,
    pub response: f64,
    pub memory_gate_enabled: bool,
    pub memory_gate_bias: f64,
    pub memory_gate_response: f64,
    pub incoming: Vec<PolicyIncomingEdge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPolicySpec {
    pub network_type: String,
    pub input_count: usize,
    pub output_indices: Vec<usize>,
    pub node_evals: Vec<CompiledPolicyNodeEval>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPolicySnapshot {
    pub node_values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPolicyRequest {
    pub inputs: Vec<Vec<f64>>,
    pub snapshot: Option<CompiledPolicySnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPolicyResult {
    pub outputs: Vec<Vec<f64>>,
    pub snapshots: Option<Vec<Option<CompiledPolicySnapshot>>>,
    pub backend_used: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyBridgeError {
    Protocol(String),
    InvalidNetworkType(String),
    InputCountMismatch {
        expected: usize,
        actual: usize,
    },
    OutputIndexOutOfRange {
        output_index: usize,
        node_count: usize,
    },
    IncomingSourceOutOfRange {
        node_id: i64,
        source_kind: String,
        source_index: usize,
    },
    UnsupportedActivation(String),
    UnsupportedAggregation(String),
    Native(String),
}

pub fn native_policy_cuda_available() -> bool {
    policy_gpu_native::native_cuda_available()
}

pub fn evaluate_policy_batch(
    spec: &CompiledPolicySpec,
    request: &CompiledPolicyRequest,
    backend: PolicyBridgeBackend,
) -> Result<CompiledPolicyResult, PolicyBridgeError> {
    spec.validate()?;
    request.validate_inputs(spec.input_count)?;

    match backend {
        PolicyBridgeBackend::Cpu => evaluate_policy_batch_cpu(spec, request, "cpu"),
        PolicyBridgeBackend::CudaNative => {
            policy_gpu_native::evaluate_policy_batch_native(spec, request)
                .map_err(PolicyBridgeError::Native)
        }
        PolicyBridgeBackend::Auto => {
            if native_policy_cuda_available() {
                if let Ok(result) = policy_gpu_native::evaluate_policy_batch_native(spec, request) {
                    return Ok(result);
                }
            }
            evaluate_policy_batch_cpu(spec, request, "cpu")
        }
    }
}

impl CompiledPolicySpec {
    pub fn from_json(value: &JsonValue) -> Result<Self, PolicyBridgeError> {
        let network_type = json_string_field(value, "networkType")?;
        let input_count = json_usize_field(value, "inputCount")?;
        let output_indices = json_usize_array_field(value, "outputIndices")?;
        let mut node_evals = Vec::new();
        let nodes_value = json_field(value, "nodeEvals")?;
        if !nodes_value.is_array() {
            return Err(PolicyBridgeError::Protocol(
                "nodeEvals must be a JSON array".to_string(),
            ));
        }
        for node in nodes_value.members() {
            node_evals.push(CompiledPolicyNodeEval::from_json(node)?);
        }
        Ok(Self {
            network_type,
            input_count,
            output_indices,
            node_evals,
        })
    }

    pub fn is_feedforward(&self) -> bool {
        self.network_type.trim().eq_ignore_ascii_case("feedforward")
    }

    pub fn is_recurrent(&self) -> bool {
        self.network_type.trim().eq_ignore_ascii_case("recurrent")
    }

    fn validate(&self) -> Result<(), PolicyBridgeError> {
        if !self.is_feedforward() && !self.is_recurrent() {
            return Err(PolicyBridgeError::InvalidNetworkType(
                self.network_type.clone(),
            ));
        }

        for &output_index in &self.output_indices {
            if output_index >= self.node_evals.len() {
                return Err(PolicyBridgeError::OutputIndexOutOfRange {
                    output_index,
                    node_count: self.node_evals.len(),
                });
            }
        }

        for node in &self.node_evals {
            normalize_js_activation(&node.activation)?;
            normalize_js_aggregation(&node.aggregation)?;
            for edge in &node.incoming {
                match normalize_source_kind(&edge.source_kind)? {
                    SourceKind::Input => {
                        if edge.source_index >= self.input_count {
                            return Err(PolicyBridgeError::IncomingSourceOutOfRange {
                                node_id: node.node_id,
                                source_kind: edge.source_kind.clone(),
                                source_index: edge.source_index,
                            });
                        }
                    }
                    SourceKind::Node => {
                        if edge.source_index >= self.node_evals.len() {
                            return Err(PolicyBridgeError::IncomingSourceOutOfRange {
                                node_id: node.node_id,
                                source_kind: edge.source_kind.clone(),
                                source_index: edge.source_index,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl CompiledPolicyRequest {
    pub fn from_json(value: &JsonValue) -> Result<Self, PolicyBridgeError> {
        let inputs_value = json_field(value, "inputs")?;
        if !inputs_value.is_array() {
            return Err(PolicyBridgeError::Protocol(
                "inputs must be a JSON array".to_string(),
            ));
        }
        let mut inputs = Vec::new();
        for row in inputs_value.members() {
            if !row.is_array() {
                return Err(PolicyBridgeError::Protocol(
                    "input rows must be JSON arrays".to_string(),
                ));
            }
            let mut values = Vec::new();
            for item in row.members() {
                values.push(json_number(item, "input value")?);
            }
            inputs.push(values);
        }

        let snapshot = if value.has_key("snapshot") && !value["snapshot"].is_null() {
            Some(CompiledPolicySnapshot::from_json(&value["snapshot"])?)
        } else {
            None
        };
        Ok(Self { inputs, snapshot })
    }

    fn validate_inputs(&self, expected: usize) -> Result<(), PolicyBridgeError> {
        for row in &self.inputs {
            if row.len() != expected {
                return Err(PolicyBridgeError::InputCountMismatch {
                    expected,
                    actual: row.len(),
                });
            }
        }
        Ok(())
    }
}

impl CompiledPolicyNodeEval {
    pub fn from_json(value: &JsonValue) -> Result<Self, PolicyBridgeError> {
        let node_id = json_i64_field(value, "nodeId")?;
        let activation = json_string_field(value, "activation")?;
        let aggregation = json_string_field(value, "aggregation")?;
        let bias = json_f64_field(value, "bias")?;
        let response = json_f64_field(value, "response")?;
        let memory_gate_enabled = json_bool_field(value, "memoryGateEnabled")?;
        let memory_gate_bias = json_f64_field(value, "memoryGateBias")?;
        let memory_gate_response = json_f64_field(value, "memoryGateResponse")?;
        let incoming_value = json_field(value, "incoming")?;
        if !incoming_value.is_array() {
            return Err(PolicyBridgeError::Protocol(
                "incoming must be a JSON array".to_string(),
            ));
        }
        let mut incoming = Vec::new();
        for edge in incoming_value.members() {
            incoming.push(PolicyIncomingEdge::from_json(edge)?);
        }
        Ok(Self {
            node_id,
            activation,
            aggregation,
            bias,
            response,
            memory_gate_enabled,
            memory_gate_bias,
            memory_gate_response,
            incoming,
        })
    }
}

impl PolicyIncomingEdge {
    pub fn from_json(value: &JsonValue) -> Result<Self, PolicyBridgeError> {
        Ok(Self {
            source_kind: json_string_field(value, "sourceKind")?,
            source_index: json_usize_field(value, "sourceIndex")?,
            weight: json_f64_field(value, "weight")?,
        })
    }
}

impl CompiledPolicySnapshot {
    pub fn from_json(value: &JsonValue) -> Result<Self, PolicyBridgeError> {
        let node_values_value = json_field(value, "nodeValues")?;
        if !node_values_value.is_object() {
            return Err(PolicyBridgeError::Protocol(
                "snapshot.nodeValues must be a JSON object".to_string(),
            ));
        }
        let mut node_values = BTreeMap::new();
        for (key, item) in node_values_value.entries() {
            node_values.insert(key.to_string(), json_number(item, "snapshot node value")?);
        }
        Ok(Self { node_values })
    }

    pub fn to_json_value(&self) -> JsonValue {
        let mut node_values = JsonValue::new_object();
        for (key, value) in &self.node_values {
            let _ = node_values.insert(key, *value);
        }
        let mut out = JsonValue::new_object();
        out["nodeValues"] = node_values;
        out
    }
}

impl CompiledPolicyResult {
    pub fn to_json_value(&self) -> JsonValue {
        let mut outputs = JsonValue::new_array();
        for row in &self.outputs {
            let mut row_value = JsonValue::new_array();
            for item in row {
                let _ = row_value.push(*item);
            }
            let _ = outputs.push(row_value);
        }
        let mut out = JsonValue::new_object();
        out["outputs"] = outputs;
        if let Some(snapshots) = &self.snapshots {
            let mut values = JsonValue::new_array();
            for snapshot in snapshots {
                let _ = values.push(
                    snapshot
                        .as_ref()
                        .map(CompiledPolicySnapshot::to_json_value)
                        .unwrap_or(JsonValue::Null),
                );
            }
            out["snapshots"] = values;
        }
        out["backendUsed"] = self.backend_used.as_str().into();
        out
    }
}

fn evaluate_policy_batch_cpu(
    spec: &CompiledPolicySpec,
    request: &CompiledPolicyRequest,
    backend_used: &str,
) -> Result<CompiledPolicyResult, PolicyBridgeError> {
    let mut outputs = Vec::with_capacity(request.inputs.len());
    let mut snapshots = if spec.is_recurrent() {
        Some(Vec::with_capacity(request.inputs.len()))
    } else {
        None
    };
    let base_snapshot = recurrent_snapshot_to_dense(spec, request.snapshot.as_ref());

    for input_row in &request.inputs {
        if spec.is_recurrent() {
            let (row_outputs, next_snapshot) =
                evaluate_recurrent_cpu(spec, input_row, &base_snapshot)?;
            outputs.push(row_outputs);
            if let Some(all) = snapshots.as_mut() {
                all.push(Some(dense_snapshot_to_json(spec, &next_snapshot)));
            }
        } else {
            outputs.push(evaluate_feedforward_cpu(spec, input_row)?);
        }
    }

    Ok(CompiledPolicyResult {
        outputs,
        snapshots,
        backend_used: backend_used.to_string(),
    })
}

fn evaluate_feedforward_cpu(
    spec: &CompiledPolicySpec,
    input_row: &[f64],
) -> Result<Vec<f64>, PolicyBridgeError> {
    let mut node_values = vec![0.0; spec.node_evals.len()];
    let mut terms = Vec::new();

    for (node_index, node) in spec.node_evals.iter().enumerate() {
        terms.clear();
        for edge in &node.incoming {
            let source_value = match normalize_source_kind(&edge.source_kind)? {
                SourceKind::Input => input_row[edge.source_index],
                SourceKind::Node => node_values[edge.source_index],
            };
            terms.push(source_value * edge.weight);
        }
        let aggregated = apply_js_aggregation(&node.aggregation, &terms)?;
        let pre = node.bias + (node.response * aggregated);
        node_values[node_index] = apply_js_activation(&node.activation, pre)?;
    }

    Ok(spec
        .output_indices
        .iter()
        .map(|&index| node_values[index])
        .collect())
}

fn evaluate_recurrent_cpu(
    spec: &CompiledPolicySpec,
    input_row: &[f64],
    base_snapshot: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PolicyBridgeError> {
    let mut next_values = vec![0.0; spec.node_evals.len()];
    let mut terms = Vec::new();

    for (node_index, node) in spec.node_evals.iter().enumerate() {
        terms.clear();
        for edge in &node.incoming {
            let source_value = match normalize_source_kind(&edge.source_kind)? {
                SourceKind::Input => input_row[edge.source_index],
                SourceKind::Node => base_snapshot[edge.source_index],
            };
            terms.push(source_value * edge.weight);
        }
        let aggregated = apply_js_aggregation(&node.aggregation, &terms)?;
        let candidate_pre = node.bias + (node.response * aggregated);
        let candidate_value = apply_js_activation(&node.activation, candidate_pre)?;
        next_values[node_index] = if node.memory_gate_enabled {
            let gate_pre = node.memory_gate_bias + (node.memory_gate_response * aggregated);
            let gate = sigmoid(gate_pre);
            ((1.0 - gate) * base_snapshot[node_index]) + (gate * candidate_value)
        } else {
            candidate_value
        };
    }

    let outputs = spec
        .output_indices
        .iter()
        .map(|&index| next_values[index])
        .collect();
    Ok((outputs, next_values))
}

fn recurrent_snapshot_to_dense(
    spec: &CompiledPolicySpec,
    snapshot: Option<&CompiledPolicySnapshot>,
) -> Vec<f64> {
    let mut dense = vec![0.0; spec.node_evals.len()];
    let Some(snapshot) = snapshot else {
        return dense;
    };
    for (index, node) in spec.node_evals.iter().enumerate() {
        if let Some(value) = snapshot.node_values.get(&node.node_id.to_string()) {
            dense[index] = *value;
        }
    }
    dense
}

fn dense_snapshot_to_json(spec: &CompiledPolicySpec, values: &[f64]) -> CompiledPolicySnapshot {
    let mut node_values = BTreeMap::new();
    for (index, node) in spec.node_evals.iter().enumerate() {
        node_values.insert(
            node.node_id.to_string(),
            values.get(index).copied().unwrap_or(0.0),
        );
    }
    CompiledPolicySnapshot { node_values }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Input,
    Node,
}

fn normalize_source_kind(value: &str) -> Result<SourceKind, PolicyBridgeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "input" => Ok(SourceKind::Input),
        "node" => Ok(SourceKind::Node),
        other => Err(PolicyBridgeError::Protocol(format!(
            "unsupported source kind '{other}'"
        ))),
    }
}

fn normalize_js_activation(value: &str) -> Result<&str, PolicyBridgeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "sigmoid" => Ok("sigmoid"),
        "tanh" => Ok("tanh"),
        "relu" => Ok("relu"),
        "identity" | "linear" => Ok("identity"),
        "clamped" => Ok("clamped"),
        "gauss" => Ok("gauss"),
        "sin" => Ok("sin"),
        "abs" => Ok("abs"),
        other => Err(PolicyBridgeError::UnsupportedActivation(other.to_string())),
    }
}

fn normalize_js_aggregation(value: &str) -> Result<&str, PolicyBridgeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "sum" => Ok("sum"),
        "mean" => Ok("mean"),
        "max" => Ok("max"),
        "min" => Ok("min"),
        "product" => Ok("product"),
        "maxabs" => Ok("maxabs"),
        other => Err(PolicyBridgeError::UnsupportedAggregation(other.to_string())),
    }
}

fn apply_js_activation(name: &str, value: f64) -> Result<f64, PolicyBridgeError> {
    Ok(match normalize_js_activation(name)? {
        "sigmoid" => sigmoid(value),
        "tanh" => value.tanh(),
        "relu" => value.max(0.0),
        "identity" => value,
        "clamped" => value.clamp(-1.0, 1.0),
        "gauss" => (-(value * value)).exp(),
        "sin" => value.sin(),
        "abs" => value.abs(),
        _ => value,
    })
}

fn apply_js_aggregation(name: &str, values: &[f64]) -> Result<f64, PolicyBridgeError> {
    Ok(match normalize_js_aggregation(name)? {
        "sum" => values.iter().sum(),
        "mean" => {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        }
        "max" => values.iter().copied().reduce(f64::max).unwrap_or(0.0),
        "min" => values.iter().copied().reduce(f64::min).unwrap_or(0.0),
        "product" => values.iter().product(),
        "maxabs" => {
            let Some(first) = values.first().copied() else {
                return Ok(0.0);
            };
            let mut best = first;
            for value in values.iter().copied().skip(1) {
                if value.abs() > best.abs() {
                    best = value;
                }
            }
            best
        }
        _ => 0.0,
    })
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn json_field<'a>(value: &'a JsonValue, key: &str) -> Result<&'a JsonValue, PolicyBridgeError> {
    if !value.is_object() {
        return Err(PolicyBridgeError::Protocol(format!(
            "expected JSON object containing key {key:?}"
        )));
    }
    let field = &value[key];
    if field.is_null() {
        Err(PolicyBridgeError::Protocol(format!(
            "missing required JSON field {key:?}"
        )))
    } else {
        Ok(field)
    }
}

fn json_string_field(value: &JsonValue, key: &str) -> Result<String, PolicyBridgeError> {
    json_field(value, key)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| PolicyBridgeError::Protocol(format!("JSON field {key:?} must be a string")))
}

fn json_bool_field(value: &JsonValue, key: &str) -> Result<bool, PolicyBridgeError> {
    json_field(value, key)?
        .as_bool()
        .ok_or_else(|| PolicyBridgeError::Protocol(format!("JSON field {key:?} must be a bool")))
}

fn json_number(value: &JsonValue, label: &str) -> Result<f64, PolicyBridgeError> {
    value
        .as_f64()
        .ok_or_else(|| PolicyBridgeError::Protocol(format!("{label} must be numeric")))
}

fn json_f64_field(value: &JsonValue, key: &str) -> Result<f64, PolicyBridgeError> {
    json_number(json_field(value, key)?, &format!("JSON field {key:?}"))
}

fn json_usize_field(value: &JsonValue, key: &str) -> Result<usize, PolicyBridgeError> {
    json_field(value, key)?
        .as_usize()
        .ok_or_else(|| PolicyBridgeError::Protocol(format!("JSON field {key:?} must be a usize")))
}

fn json_i64_field(value: &JsonValue, key: &str) -> Result<i64, PolicyBridgeError> {
    json_field(value, key)?
        .as_i64()
        .ok_or_else(|| PolicyBridgeError::Protocol(format!("JSON field {key:?} must be an i64")))
}

fn json_usize_array_field(value: &JsonValue, key: &str) -> Result<Vec<usize>, PolicyBridgeError> {
    let array = json_field(value, key)?;
    if !array.is_array() {
        return Err(PolicyBridgeError::Protocol(format!(
            "JSON field {key:?} must be an array"
        )));
    }
    let mut out = Vec::new();
    for item in array.members() {
        out.push(item.as_usize().ok_or_else(|| {
            PolicyBridgeError::Protocol(format!(
                "JSON field {key:?} must contain only usize values"
            ))
        })?);
    }
    Ok(out)
}

impl fmt::Display for PolicyBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(message) => write!(f, "{message}"),
            Self::InvalidNetworkType(value) => {
                write!(
                    f,
                    "unsupported network_type {value:?}; expected feedforward or recurrent"
                )
            }
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::OutputIndexOutOfRange {
                output_index,
                node_count,
            } => write!(
                f,
                "output index {output_index} is out of range for node count {node_count}"
            ),
            Self::IncomingSourceOutOfRange {
                node_id,
                source_kind,
                source_index,
            } => write!(
                f,
                "node {node_id}: incoming edge source {source_kind}:{source_index} is out of range"
            ),
            Self::UnsupportedActivation(value) => {
                write!(
                    f,
                    "unsupported JS activation {value:?} for native policy bridge"
                )
            }
            Self::UnsupportedAggregation(value) => {
                write!(
                    f,
                    "unsupported JS aggregation {value:?} for native policy bridge"
                )
            }
            Self::Native(message) => write!(f, "{message}"),
        }
    }
}

impl Error for PolicyBridgeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn feedforward_fixture() -> CompiledPolicySpec {
        CompiledPolicySpec {
            network_type: "feedforward".to_string(),
            input_count: 2,
            output_indices: vec![0],
            node_evals: vec![CompiledPolicyNodeEval {
                node_id: 0,
                activation: "identity".to_string(),
                aggregation: "sum".to_string(),
                bias: 0.5,
                response: 1.0,
                memory_gate_enabled: false,
                memory_gate_bias: 0.0,
                memory_gate_response: 1.0,
                incoming: vec![
                    PolicyIncomingEdge {
                        source_kind: "input".to_string(),
                        source_index: 0,
                        weight: 2.0,
                    },
                    PolicyIncomingEdge {
                        source_kind: "input".to_string(),
                        source_index: 1,
                        weight: -1.0,
                    },
                ],
            }],
        }
    }

    #[test]
    fn evaluates_feedforward_fixture_on_cpu() {
        let result = evaluate_policy_batch(
            &feedforward_fixture(),
            &CompiledPolicyRequest {
                inputs: vec![vec![3.0, 1.0]],
                snapshot: None,
            },
            PolicyBridgeBackend::Cpu,
        )
        .expect("policy evaluation should succeed");

        assert_eq!(result.backend_used, "cpu");
        assert_eq!(result.outputs.len(), 1);
        assert!((result.outputs[0][0] - 5.5).abs() < 1e-12);
        assert!(result.snapshots.is_none());
    }

    #[test]
    fn evaluates_recurrent_memory_gate_fixture_on_cpu() {
        let spec = CompiledPolicySpec {
            network_type: "recurrent".to_string(),
            input_count: 1,
            output_indices: vec![0],
            node_evals: vec![CompiledPolicyNodeEval {
                node_id: 0,
                activation: "identity".to_string(),
                aggregation: "sum".to_string(),
                bias: 0.0,
                response: 1.0,
                memory_gate_enabled: true,
                memory_gate_bias: 0.0,
                memory_gate_response: 1.0,
                incoming: vec![
                    PolicyIncomingEdge {
                        source_kind: "input".to_string(),
                        source_index: 0,
                        weight: 1.0,
                    },
                    PolicyIncomingEdge {
                        source_kind: "node".to_string(),
                        source_index: 0,
                        weight: 1.0,
                    },
                ],
            }],
        };
        let mut snapshot_values = BTreeMap::new();
        snapshot_values.insert("0".to_string(), 0.25);
        let result = evaluate_policy_batch(
            &spec,
            &CompiledPolicyRequest {
                inputs: vec![vec![1.0]],
                snapshot: Some(CompiledPolicySnapshot {
                    node_values: snapshot_values,
                }),
            },
            PolicyBridgeBackend::Cpu,
        )
        .expect("policy evaluation should succeed");

        let expected_gate = sigmoid(1.25);
        let expected_value = ((1.0 - expected_gate) * 0.25) + (expected_gate * 1.25);
        assert!((result.outputs[0][0] - expected_value).abs() < 1e-12);
        let snapshots = result
            .snapshots
            .expect("recurrent result should include snapshots");
        let next = snapshots[0]
            .as_ref()
            .expect("recurrent sample should include snapshot");
        assert!((next.node_values["0"] - expected_value).abs() < 1e-12);
    }
}
