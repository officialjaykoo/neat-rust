use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::native::policy_gpu;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyRuntimeBackend {
    Cpu,
    CudaNative,
}

impl PolicyRuntimeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CudaNative => "cuda_native",
        }
    }
}

impl Serialize for PolicyRuntimeBackend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PolicyRuntimeBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "rust_cpu" => Ok(Self::Cpu),
            "cuda" | "cuda_native" | "native_cuda" | "rust_cuda" => Ok(Self::CudaNative),
            other => Err(de::Error::custom(format!(
                "unsupported runtime backend {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyNetworkType {
    FeedForward,
    Recurrent,
}

impl PolicyNetworkType {
    pub fn parse(value: &str) -> Result<Self, PolicyBridgeError> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "feedforward" | "feed_forward" | "ff" => Ok(Self::FeedForward),
            "recurrent" | "rnn" => Ok(Self::Recurrent),
            other => Err(PolicyBridgeError::InvalidNetworkType(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FeedForward => "feedforward",
            Self::Recurrent => "recurrent",
        }
    }

    pub fn is_feedforward(self) -> bool {
        matches!(self, Self::FeedForward)
    }

    pub fn is_recurrent(self) -> bool {
        matches!(self, Self::Recurrent)
    }
}

impl Serialize for PolicyNetworkType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PolicyNetworkType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

impl fmt::Display for PolicyNetworkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyIncomingSource {
    Input,
    Node,
}

impl PolicyIncomingSource {
    pub fn parse(value: &str) -> Result<Self, PolicyBridgeError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "input" => Ok(Self::Input),
            "node" => Ok(Self::Node),
            other => Err(PolicyBridgeError::Protocol(format!(
                "unsupported source kind '{other}'"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Node => "node",
        }
    }
}

impl Serialize for PolicyIncomingSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PolicyIncomingSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

impl fmt::Display for PolicyIncomingSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyActivation {
    Sigmoid,
    #[default]
    Tanh,
    Relu,
    Identity,
    Clamped,
    Gauss,
    Sin,
    Abs,
}

impl PolicyActivation {
    pub fn parse(value: &str) -> Result<Self, PolicyBridgeError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "identity" | "linear" => Ok(Self::Identity),
            "clamped" => Ok(Self::Clamped),
            "gauss" => Ok(Self::Gauss),
            "sin" => Ok(Self::Sin),
            "abs" => Ok(Self::Abs),
            other => Err(PolicyBridgeError::UnsupportedActivation(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Relu => "relu",
            Self::Identity => "identity",
            Self::Clamped => "clamped",
            Self::Gauss => "gauss",
            Self::Sin => "sin",
            Self::Abs => "abs",
        }
    }

    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid(value),
            Self::Tanh => value.tanh(),
            Self::Relu => value.max(0.0),
            Self::Identity => value,
            Self::Clamped => value.clamp(-1.0, 1.0),
            Self::Gauss => (-(value * value)).exp(),
            Self::Sin => value.sin(),
            Self::Abs => value.abs(),
        }
    }

    #[cfg(windows)]
    pub(crate) fn cuda_code(self) -> i32 {
        match self {
            Self::Sigmoid => 0,
            Self::Tanh => 1,
            Self::Relu => 2,
            Self::Identity => 3,
            Self::Clamped => 4,
            Self::Gauss => 5,
            Self::Sin => 6,
            Self::Abs => 7,
        }
    }
}

impl Serialize for PolicyActivation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PolicyActivation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyAggregation {
    #[default]
    Sum,
    Mean,
    Max,
    Min,
    Product,
    MaxAbs,
}

impl PolicyAggregation {
    pub fn parse(value: &str) -> Result<Self, PolicyBridgeError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            "product" => Ok(Self::Product),
            "maxabs" => Ok(Self::MaxAbs),
            other => Err(PolicyBridgeError::UnsupportedAggregation(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
            Self::Product => "product",
            Self::MaxAbs => "maxabs",
        }
    }

    pub fn apply(self, values: &[f64]) -> f64 {
        match self {
            Self::Sum => values.iter().sum(),
            Self::Mean => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            Self::Max => values.iter().copied().reduce(f64::max).unwrap_or(0.0),
            Self::Min => values.iter().copied().reduce(f64::min).unwrap_or(0.0),
            Self::Product => values.iter().product(),
            Self::MaxAbs => {
                let Some(first) = values.first().copied() else {
                    return 0.0;
                };
                let mut best = first;
                for value in values.iter().copied().skip(1) {
                    if value.abs() > best.abs() {
                        best = value;
                    }
                }
                best
            }
        }
    }
}

impl Serialize for PolicyAggregation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PolicyAggregation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyIncomingEdge {
    pub source_kind: PolicyIncomingSource,
    pub source_index: usize,
    pub weight: f64,
    #[serde(default)]
    pub connection_gru_enabled: bool,
    #[serde(default)]
    pub connection_memory_weight: f64,
    #[serde(default)]
    pub connection_reset_input_weight: f64,
    #[serde(default)]
    pub connection_reset_memory_weight: f64,
    #[serde(default)]
    pub connection_update_input_weight: f64,
    #[serde(default)]
    pub connection_update_memory_weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledPolicyNodeEval {
    pub node_id: i64,
    #[serde(default)]
    pub activation: PolicyActivation,
    #[serde(default)]
    pub aggregation: PolicyAggregation,
    pub bias: f64,
    pub response: f64,
    #[serde(default)]
    pub memory_gate_enabled: bool,
    #[serde(default)]
    pub memory_gate_bias: f64,
    #[serde(default = "default_memory_gate_response")]
    pub memory_gate_response: f64,
    #[serde(default)]
    pub incoming: Vec<PolicyIncomingEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledPolicySpec {
    pub network_type: PolicyNetworkType,
    pub input_count: usize,
    pub output_indices: Vec<usize>,
    pub node_evals: Vec<CompiledPolicyNodeEval>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledPolicySnapshot {
    #[serde(default)]
    pub node_values: BTreeMap<String, f64>,
    #[serde(default)]
    pub connection_memory: BTreeMap<String, f64>,
    #[serde(default)]
    pub connection_prev_input: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledPolicyRequest {
    pub inputs: Vec<Vec<f64>>,
    #[serde(default)]
    pub snapshot: Option<CompiledPolicySnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledPolicyResult {
    pub outputs: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshots: Option<Vec<Option<CompiledPolicySnapshot>>>,
    pub backend_used: PolicyRuntimeBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyNativeError {
    UnsupportedPlatform,
    PtxCompilerNotFound,
    PtxCompileFailed {
        compiler: String,
        stderr: String,
    },
    PtxProcessLaunch {
        compiler: String,
        message: String,
    },
    PtxFileIo {
        operation: &'static str,
        path: String,
        message: String,
    },
    PtxContainsNul,
    PtxEmpty,
    DriverLibraryUnavailable(&'static str),
    DriverSymbolMissing(&'static str),
    DriverUnavailable(&'static str),
    CudaDriver {
        operation: &'static str,
        code: i32,
    },
    NoCudaDevice,
    SharedMemoryExceeded {
        required: usize,
        limit: usize,
    },
    UnsupportedAggregation {
        node_id: i64,
        aggregation: String,
    },
    ConnectionGruUnsupported {
        node_id: i64,
    },
    BufferOverflow(&'static str),
    MissingRecurrentBuffer(&'static str),
    InputRowSizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl From<policy_gpu::NativePolicyGpuError> for PolicyNativeError {
    fn from(value: policy_gpu::NativePolicyGpuError) -> Self {
        match value {
            policy_gpu::NativePolicyGpuError::UnsupportedPlatform => Self::UnsupportedPlatform,
            policy_gpu::NativePolicyGpuError::PtxCompilerNotFound => Self::PtxCompilerNotFound,
            policy_gpu::NativePolicyGpuError::PtxCompileFailed { compiler, stderr } => {
                Self::PtxCompileFailed { compiler, stderr }
            }
            policy_gpu::NativePolicyGpuError::PtxProcessLaunch { compiler, message } => {
                Self::PtxProcessLaunch { compiler, message }
            }
            policy_gpu::NativePolicyGpuError::PtxFileIo {
                operation,
                path,
                message,
            } => Self::PtxFileIo {
                operation,
                path,
                message,
            },
            policy_gpu::NativePolicyGpuError::PtxContainsNul => Self::PtxContainsNul,
            policy_gpu::NativePolicyGpuError::PtxEmpty => Self::PtxEmpty,
            policy_gpu::NativePolicyGpuError::DriverLibraryUnavailable(name) => {
                Self::DriverLibraryUnavailable(name)
            }
            policy_gpu::NativePolicyGpuError::DriverSymbolMissing(name) => {
                Self::DriverSymbolMissing(name)
            }
            policy_gpu::NativePolicyGpuError::DriverUnavailable(operation) => {
                Self::DriverUnavailable(operation)
            }
            policy_gpu::NativePolicyGpuError::CudaDriver { operation, code } => {
                Self::CudaDriver { operation, code }
            }
            policy_gpu::NativePolicyGpuError::NoCudaDevice => Self::NoCudaDevice,
            policy_gpu::NativePolicyGpuError::SharedMemoryExceeded { required, limit } => {
                Self::SharedMemoryExceeded { required, limit }
            }
            policy_gpu::NativePolicyGpuError::UnsupportedAggregation {
                node_id,
                aggregation,
            } => Self::UnsupportedAggregation {
                node_id,
                aggregation,
            },
            policy_gpu::NativePolicyGpuError::ConnectionGruUnsupported { node_id } => {
                Self::ConnectionGruUnsupported { node_id }
            }
            policy_gpu::NativePolicyGpuError::BufferOverflow(label) => Self::BufferOverflow(label),
            policy_gpu::NativePolicyGpuError::MissingRecurrentBuffer(label) => {
                Self::MissingRecurrentBuffer(label)
            }
            policy_gpu::NativePolicyGpuError::InputRowSizeMismatch { expected, actual } => {
                Self::InputRowSizeMismatch { expected, actual }
            }
        }
    }
}

impl fmt::Display for PolicyNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                write!(f, "native CUDA policy bridge is unavailable on this platform")
            }
            Self::PtxCompilerNotFound => write!(f, "clang++ not found for CUDA PTX compilation"),
            Self::PtxCompileFailed { compiler, stderr } => {
                write!(f, "failed to compile CUDA PTX with {compiler}: {stderr}")
            }
            Self::PtxProcessLaunch { compiler, message } => {
                write!(f, "failed to run {compiler}: {message}")
            }
            Self::PtxFileIo {
                operation,
                path,
                message,
            } => write!(f, "failed to {operation} PTX file {path}: {message}"),
            Self::PtxContainsNul => write!(f, "PTX contains NUL byte"),
            Self::PtxEmpty => write!(f, "compiled CUDA PTX was empty"),
            Self::DriverLibraryUnavailable(name) => write!(f, "{name} is unavailable"),
            Self::DriverSymbolMissing(name) => write!(f, "{name} symbol is unavailable"),
            Self::DriverUnavailable(operation) => write!(f, "{operation} is unavailable"),
            Self::CudaDriver { operation, code } => {
                write!(f, "{operation} failed with CUDA error {code}")
            }
            Self::NoCudaDevice => write!(f, "no CUDA device found"),
            Self::SharedMemoryExceeded { required, limit } => write!(
                f,
                "policy CUDA kernel requires {required} bytes shared memory; limit is {limit}"
            ),
            Self::UnsupportedAggregation {
                node_id,
                aggregation,
            } => write!(
                f,
                "native policy CUDA currently supports only sum aggregation; node {node_id} uses {aggregation}"
            ),
            Self::ConnectionGruUnsupported { node_id } => write!(
                f,
                "native policy CUDA does not support connection-GRU edges yet; node {node_id} uses one"
            ),
            Self::BufferOverflow(label) => write!(f, "{label} size overflow"),
            Self::MissingRecurrentBuffer(label) => {
                write!(f, "missing recurrent {label} device buffer")
            }
            Self::InputRowSizeMismatch { expected, actual } => write!(
                f,
                "policy input row size mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for PolicyNativeError {}

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
        source_kind: PolicyIncomingSource,
        source_index: usize,
    },
    UnsupportedActivation(String),
    UnsupportedAggregation(String),
    Native(PolicyNativeError),
}

impl From<policy_gpu::NativePolicyGpuError> for PolicyBridgeError {
    fn from(value: policy_gpu::NativePolicyGpuError) -> Self {
        Self::Native(value.into())
    }
}

pub fn native_policy_cuda_available() -> bool {
    policy_gpu::native_cuda_available()
}

pub fn evaluate_policy_batch(
    spec: &CompiledPolicySpec,
    request: &CompiledPolicyRequest,
    backend: PolicyBridgeBackend,
) -> Result<CompiledPolicyResult, PolicyBridgeError> {
    match backend {
        PolicyBridgeBackend::Cpu => CpuPolicyEvaluator.evaluate(spec, request),
        PolicyBridgeBackend::CudaNative => CudaNativePolicyEvaluator.evaluate(spec, request),
        PolicyBridgeBackend::Auto => AutoPolicyEvaluator.evaluate(spec, request),
    }
}

pub trait PolicyBatchEvaluator {
    fn evaluate(
        &self,
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, PolicyBridgeError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuPolicyEvaluator;

impl PolicyBatchEvaluator for CpuPolicyEvaluator {
    fn evaluate(
        &self,
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, PolicyBridgeError> {
        spec.validate()?;
        request.validate_inputs(spec.input_count)?;
        evaluate_policy_batch_cpu(spec, request, PolicyRuntimeBackend::Cpu)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CudaNativePolicyEvaluator;

impl PolicyBatchEvaluator for CudaNativePolicyEvaluator {
    fn evaluate(
        &self,
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, PolicyBridgeError> {
        spec.validate()?;
        request.validate_inputs(spec.input_count)?;
        policy_gpu::evaluate_policy_batch_native(spec, request).map_err(PolicyBridgeError::from)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AutoPolicyEvaluator;

impl PolicyBatchEvaluator for AutoPolicyEvaluator {
    fn evaluate(
        &self,
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, PolicyBridgeError> {
        spec.validate()?;
        request.validate_inputs(spec.input_count)?;
        if native_policy_cuda_available() {
            if let Ok(result) = policy_gpu::evaluate_policy_batch_native(spec, request) {
                return Ok(result);
            }
        }
        evaluate_policy_batch_cpu(spec, request, PolicyRuntimeBackend::Cpu)
    }
}

impl CompiledPolicySpec {
    pub fn from_json_text(text: &str) -> Result<Self, PolicyBridgeError> {
        serde_json::from_str(text).map_err(|err| {
            PolicyBridgeError::Protocol(format!("invalid compiled policy JSON: {err}"))
        })
    }

    pub fn is_feedforward(&self) -> bool {
        self.network_type.is_feedforward()
    }

    pub fn is_recurrent(&self) -> bool {
        self.network_type.is_recurrent()
    }

    pub fn validate(&self) -> Result<(), PolicyBridgeError> {
        for &output_index in &self.output_indices {
            if output_index >= self.node_evals.len() {
                return Err(PolicyBridgeError::OutputIndexOutOfRange {
                    output_index,
                    node_count: self.node_evals.len(),
                });
            }
        }

        for node in &self.node_evals {
            for edge in &node.incoming {
                match edge.source_kind {
                    PolicyIncomingSource::Input => {
                        if edge.source_index >= self.input_count {
                            return Err(PolicyBridgeError::IncomingSourceOutOfRange {
                                node_id: node.node_id,
                                source_kind: edge.source_kind,
                                source_index: edge.source_index,
                            });
                        }
                    }
                    PolicyIncomingSource::Node => {
                        if edge.source_index >= self.node_evals.len() {
                            return Err(PolicyBridgeError::IncomingSourceOutOfRange {
                                node_id: node.node_id,
                                source_kind: edge.source_kind,
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
    pub fn from_json_text(text: &str) -> Result<Self, PolicyBridgeError> {
        serde_json::from_str(text).map_err(|err| {
            PolicyBridgeError::Protocol(format!("invalid policy request JSON: {err}"))
        })
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

fn evaluate_policy_batch_cpu(
    spec: &CompiledPolicySpec,
    request: &CompiledPolicyRequest,
    backend_used: PolicyRuntimeBackend,
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
                evaluate_recurrent_cpu(spec, input_row, &base_snapshot, request.snapshot.as_ref());
            outputs.push(row_outputs);
            if let Some(all) = snapshots.as_mut() {
                all.push(Some(next_snapshot));
            }
        } else {
            outputs.push(evaluate_feedforward_cpu(spec, input_row));
        }
    }

    Ok(CompiledPolicyResult {
        outputs,
        snapshots,
        backend_used,
    })
}

fn evaluate_feedforward_cpu(spec: &CompiledPolicySpec, input_row: &[f64]) -> Vec<f64> {
    let mut node_values = vec![0.0; spec.node_evals.len()];
    let mut terms = Vec::new();

    for (node_index, node) in spec.node_evals.iter().enumerate() {
        terms.clear();
        for edge in &node.incoming {
            let source_value = match edge.source_kind {
                PolicyIncomingSource::Input => input_row[edge.source_index],
                PolicyIncomingSource::Node => node_values[edge.source_index],
            };
            terms.push(source_value * edge.weight);
        }
        let aggregated = node.aggregation.apply(&terms);
        let pre = node.bias + (node.response * aggregated);
        node_values[node_index] = node.activation.apply(pre);
    }

    spec.output_indices
        .iter()
        .map(|&index| node_values[index])
        .collect()
}

fn evaluate_recurrent_cpu(
    spec: &CompiledPolicySpec,
    input_row: &[f64],
    base_snapshot: &[f64],
    snapshot: Option<&CompiledPolicySnapshot>,
) -> (Vec<f64>, CompiledPolicySnapshot) {
    let mut next_values = vec![0.0; spec.node_evals.len()];
    let mut connection_memory = BTreeMap::new();
    let mut connection_prev_input = BTreeMap::new();
    let mut terms = Vec::new();

    for (node_index, node) in spec.node_evals.iter().enumerate() {
        terms.clear();
        for (edge_index, edge) in node.incoming.iter().enumerate() {
            let source_value = match edge.source_kind {
                PolicyIncomingSource::Input => input_row[edge.source_index],
                PolicyIncomingSource::Node => base_snapshot[edge.source_index],
            };
            let contribution = if edge.connection_gru_enabled {
                let state_key = connection_state_key(node.node_id, edge_index);
                let previous_memory = snapshot
                    .and_then(|snapshot| snapshot.connection_memory.get(&state_key))
                    .copied()
                    .unwrap_or(0.0);
                let previous_input = snapshot
                    .and_then(|snapshot| snapshot.connection_prev_input.get(&state_key))
                    .copied()
                    .unwrap_or(0.0);
                let reset_gate = sigmoid(
                    (source_value * edge.connection_reset_input_weight)
                        + (previous_memory * edge.connection_reset_memory_weight),
                );
                let candidate_memory = ((previous_input * edge.weight)
                    + (reset_gate * previous_memory * edge.connection_memory_weight))
                    .tanh();
                let update_gate = sigmoid(
                    (source_value * edge.connection_update_input_weight)
                        + (previous_memory * edge.connection_update_memory_weight),
                );
                let next_memory =
                    (update_gate * previous_memory) + ((1.0 - update_gate) * candidate_memory);
                connection_memory.insert(state_key.clone(), next_memory);
                connection_prev_input.insert(state_key, source_value);
                (source_value * edge.weight) + (next_memory * edge.connection_memory_weight)
            } else {
                source_value * edge.weight
            };
            terms.push(contribution);
        }
        let aggregated = node.aggregation.apply(&terms);
        let candidate_pre = node.bias + (node.response * aggregated);
        let candidate_value = node.activation.apply(candidate_pre);
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
    (
        outputs,
        dense_snapshot_to_json(spec, &next_values, connection_memory, connection_prev_input),
    )
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

fn dense_snapshot_to_json(
    spec: &CompiledPolicySpec,
    values: &[f64],
    connection_memory: BTreeMap<String, f64>,
    connection_prev_input: BTreeMap<String, f64>,
) -> CompiledPolicySnapshot {
    let mut node_values = BTreeMap::new();
    for (index, node) in spec.node_evals.iter().enumerate() {
        node_values.insert(
            node.node_id.to_string(),
            values.get(index).copied().unwrap_or(0.0),
        );
    }
    CompiledPolicySnapshot {
        node_values,
        connection_memory,
        connection_prev_input,
    }
}

fn connection_state_key(node_id: i64, edge_index: usize) -> String {
    format!("{node_id}:{edge_index}")
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn default_memory_gate_response() -> f64 {
    1.0
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
            Self::Native(err) => write!(f, "{err}"),
        }
    }
}

impl Error for PolicyBridgeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_edge(
        source_kind: PolicyIncomingSource,
        source_index: usize,
        weight: f64,
    ) -> PolicyIncomingEdge {
        PolicyIncomingEdge {
            source_kind,
            source_index,
            weight,
            connection_gru_enabled: false,
            connection_memory_weight: 0.0,
            connection_reset_input_weight: 0.0,
            connection_reset_memory_weight: 0.0,
            connection_update_input_weight: 0.0,
            connection_update_memory_weight: 0.0,
        }
    }

    fn feedforward_fixture() -> CompiledPolicySpec {
        CompiledPolicySpec {
            network_type: PolicyNetworkType::FeedForward,
            input_count: 2,
            output_indices: vec![0],
            node_evals: vec![CompiledPolicyNodeEval {
                node_id: 0,
                activation: PolicyActivation::Identity,
                aggregation: PolicyAggregation::Sum,
                bias: 0.5,
                response: 1.0,
                memory_gate_enabled: false,
                memory_gate_bias: 0.0,
                memory_gate_response: 1.0,
                incoming: vec![
                    plain_edge(PolicyIncomingSource::Input, 0, 2.0),
                    plain_edge(PolicyIncomingSource::Input, 1, -1.0),
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

        assert_eq!(result.backend_used, PolicyRuntimeBackend::Cpu);
        assert_eq!(result.outputs.len(), 1);
        assert!((result.outputs[0][0] - 5.5).abs() < 1e-12);
        assert!(result.snapshots.is_none());
    }

    #[test]
    fn evaluates_recurrent_memory_gate_fixture_on_cpu() {
        let spec = CompiledPolicySpec {
            network_type: PolicyNetworkType::Recurrent,
            input_count: 1,
            output_indices: vec![0],
            node_evals: vec![CompiledPolicyNodeEval {
                node_id: 0,
                activation: PolicyActivation::Identity,
                aggregation: PolicyAggregation::Sum,
                bias: 0.0,
                response: 1.0,
                memory_gate_enabled: true,
                memory_gate_bias: 0.0,
                memory_gate_response: 1.0,
                incoming: vec![
                    plain_edge(PolicyIncomingSource::Input, 0, 1.0),
                    plain_edge(PolicyIncomingSource::Node, 0, 1.0),
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
                    ..CompiledPolicySnapshot::default()
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

    #[test]
    fn evaluates_recurrent_connection_gru_state_on_cpu() {
        let spec = CompiledPolicySpec {
            network_type: PolicyNetworkType::Recurrent,
            input_count: 1,
            output_indices: vec![0],
            node_evals: vec![CompiledPolicyNodeEval {
                node_id: 0,
                activation: PolicyActivation::Identity,
                aggregation: PolicyAggregation::Sum,
                bias: 0.0,
                response: 1.0,
                memory_gate_enabled: false,
                memory_gate_bias: 0.0,
                memory_gate_response: 1.0,
                incoming: vec![PolicyIncomingEdge {
                    connection_gru_enabled: true,
                    connection_memory_weight: 1.0,
                    ..plain_edge(PolicyIncomingSource::Input, 0, 1.0)
                }],
            }],
        };

        let first = evaluate_policy_batch(
            &spec,
            &CompiledPolicyRequest {
                inputs: vec![vec![2.0]],
                snapshot: None,
            },
            PolicyBridgeBackend::Cpu,
        )
        .expect("first recurrent policy evaluation should succeed");
        assert!((first.outputs[0][0] - 2.0).abs() < 1e-12);

        let first_snapshot = first
            .snapshots
            .and_then(|mut snapshots| snapshots.remove(0))
            .expect("first step should return connection state");
        assert_eq!(first_snapshot.connection_prev_input["0:0"], 2.0);

        let second = evaluate_policy_batch(
            &spec,
            &CompiledPolicyRequest {
                inputs: vec![vec![2.0]],
                snapshot: Some(first_snapshot),
            },
            PolicyBridgeBackend::Cpu,
        )
        .expect("second recurrent policy evaluation should succeed");
        let expected_memory = 0.5 * 2.0_f64.tanh();
        assert!((second.outputs[0][0] - (2.0 + expected_memory)).abs() < 1e-12);
    }
}
