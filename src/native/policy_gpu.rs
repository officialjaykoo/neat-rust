#[cfg(windows)]
use std::collections::BTreeMap;

#[cfg(windows)]
use crate::policy_bridge::CompiledPolicySnapshot;
use crate::policy_bridge::{CompiledPolicyRequest, CompiledPolicyResult, CompiledPolicySpec};

#[cfg(windows)]
const CUDA_KERNEL_SOURCE: &str = include_str!("../cuda_kernels.cu");

pub(crate) fn native_cuda_available() -> bool {
    imp::native_cuda_available()
}

pub(crate) fn evaluate_policy_batch_native(
    spec: &CompiledPolicySpec,
    request: &CompiledPolicyRequest,
) -> Result<CompiledPolicyResult, String> {
    imp::evaluate_policy_batch_native(spec, request)
}

#[cfg(not(windows))]
mod imp {
    use super::*;

    pub(crate) fn native_cuda_available() -> bool {
        false
    }

    pub(crate) fn evaluate_policy_batch_native(
        _spec: &CompiledPolicySpec,
        _request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, String> {
        Err("native CUDA policy bridge is only available on Windows in this build".to_string())
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::env;
    use std::ffi::{c_char, c_int, c_void, CString, OsStr};
    use std::fs;
    use std::mem;
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;
    use std::process::Command;
    use std::ptr;
    use std::sync::OnceLock;

    type CuResult = i32;
    type CuDevice = c_int;
    type CuContext = *mut c_void;
    type CuModule = *mut c_void;
    type CuFunction = *mut c_void;
    type CuStream = *mut c_void;
    type CuDevicePtr = u64;

    const CUDA_SUCCESS: CuResult = 0;
    const SHARED_LIMIT_BYTES: usize = 48 * 1024;
    static CUDA_PTX_CACHE: OnceLock<Result<String, String>> = OnceLock::new();

    type CuInit = unsafe extern "system" fn(u32) -> CuResult;
    type CuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, c_int) -> CuResult;
    type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
    type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
    type CuModuleLoadDataEx = unsafe extern "system" fn(
        *mut CuModule,
        *const c_void,
        u32,
        *mut c_int,
        *mut *mut c_void,
    ) -> CuResult;
    type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
    type CuModuleGetFunction =
        unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
    type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
    type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
    type CuMemcpyHtoD = unsafe extern "system" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
    type CuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
    type CuLaunchKernel = unsafe extern "system" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult;
    type CuCtxSynchronize = unsafe extern "system" fn() -> CuResult;

    pub(crate) fn native_cuda_available() -> bool {
        if compiled_ptx().is_err() {
            return false;
        }
        CudaDriver::load().is_ok()
    }

    pub(crate) fn evaluate_policy_batch_native(
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
    ) -> Result<CompiledPolicyResult, String> {
        ensure_supported(spec)?;
        let batch_size = request.inputs.len();
        if batch_size == 0 {
            return Ok(CompiledPolicyResult {
                outputs: Vec::new(),
                snapshots: if spec.is_recurrent() {
                    Some(Vec::new())
                } else {
                    None
                },
                backend_used: "cuda_native".to_string(),
            });
        }

        let packed = PackedPolicyModel::from_spec(spec)?;
        let inputs = flatten_inputs_as_f32(spec.input_count, &request.inputs)?;
        let base_snapshot = flatten_snapshot_as_f32(spec, request, batch_size);

        let driver = CudaDriver::load()?;
        let context = driver.create_context()?;
        let module = context.load_module(compiled_ptx()?)?;
        let kernel_name = if spec.is_recurrent() {
            b"policy_recurrent_kernel\0".as_slice()
        } else {
            b"policy_feedforward_kernel\0".as_slice()
        };
        let kernel = module.function(kernel_name)?;

        let d_activation = DeviceBuffer::from_slice(&driver, &packed.activation_codes)?;
        let d_bias = DeviceBuffer::from_slice(&driver, &packed.bias)?;
        let d_response = DeviceBuffer::from_slice(&driver, &packed.response)?;
        let d_output_indices = DeviceBuffer::from_slice(&driver, &packed.output_indices)?;
        let d_source_kind = DeviceBuffer::from_slice(&driver, &packed.source_kind)?;
        let d_source_index = DeviceBuffer::from_slice(&driver, &packed.source_index)?;
        let d_weights = DeviceBuffer::from_slice(&driver, &packed.weights)?;
        let d_incoming_offsets = DeviceBuffer::from_slice(&driver, &packed.incoming_offsets)?;
        let d_inputs = DeviceBuffer::from_slice(&driver, &inputs)?;
        let d_outputs = DeviceBuffer::zeroed::<f32>(
            &driver,
            batch_size
                .checked_mul(spec.output_indices.len())
                .ok_or_else(|| "policy output buffer size overflow".to_string())?,
        )?;

        let mut d_memory_gate_enabled = None;
        let mut d_memory_gate_bias = None;
        let mut d_memory_gate_response = None;
        let mut d_snapshots_in = None;
        let mut d_snapshots_out = None;

        if spec.is_recurrent() {
            d_memory_gate_enabled = Some(DeviceBuffer::from_slice(
                &driver,
                &packed.memory_gate_enabled,
            )?);
            d_memory_gate_bias = Some(DeviceBuffer::from_slice(&driver, &packed.memory_gate_bias)?);
            d_memory_gate_response = Some(DeviceBuffer::from_slice(
                &driver,
                &packed.memory_gate_response,
            )?);
            let snapshot_len = batch_size
                .checked_mul(spec.node_evals.len())
                .ok_or_else(|| "policy snapshot buffer size overflow".to_string())?;
            d_snapshots_in = Some(DeviceBuffer::from_slice(&driver, &base_snapshot)?);
            d_snapshots_out = Some(DeviceBuffer::zeroed::<f32>(&driver, snapshot_len)?);
        }

        let batch_size_i32 = batch_size as i32;
        let input_count_i32 = spec.input_count as i32;
        let node_count_i32 = spec.node_evals.len() as i32;
        let output_count_i32 = spec.output_indices.len() as i32;
        let shared_bytes = policy_shared_bytes(spec.node_evals.len())?;

        if spec.is_recurrent() {
            let d_memory_gate_enabled = d_memory_gate_enabled
                .as_ref()
                .ok_or_else(|| "missing recurrent gate device buffer".to_string())?;
            let d_memory_gate_bias = d_memory_gate_bias
                .as_ref()
                .ok_or_else(|| "missing recurrent gate bias device buffer".to_string())?;
            let d_memory_gate_response = d_memory_gate_response
                .as_ref()
                .ok_or_else(|| "missing recurrent gate response device buffer".to_string())?;
            let d_snapshots_in = d_snapshots_in
                .as_ref()
                .ok_or_else(|| "missing recurrent snapshot input buffer".to_string())?;
            let d_snapshots_out = d_snapshots_out
                .as_ref()
                .ok_or_else(|| "missing recurrent snapshot output buffer".to_string())?;

            let mut params = [
                ptr_to_kernel_arg(&d_activation.ptr),
                ptr_to_kernel_arg(&d_bias.ptr),
                ptr_to_kernel_arg(&d_response.ptr),
                ptr_to_kernel_arg(&d_memory_gate_enabled.ptr),
                ptr_to_kernel_arg(&d_memory_gate_bias.ptr),
                ptr_to_kernel_arg(&d_memory_gate_response.ptr),
                ptr_to_kernel_arg(&d_output_indices.ptr),
                ptr_to_kernel_arg(&d_source_kind.ptr),
                ptr_to_kernel_arg(&d_source_index.ptr),
                ptr_to_kernel_arg(&d_weights.ptr),
                ptr_to_kernel_arg(&d_incoming_offsets.ptr),
                ptr_to_kernel_arg(&batch_size_i32),
                ptr_to_kernel_arg(&input_count_i32),
                ptr_to_kernel_arg(&node_count_i32),
                ptr_to_kernel_arg(&output_count_i32),
                ptr_to_kernel_arg(&d_inputs.ptr),
                ptr_to_kernel_arg(&d_snapshots_in.ptr),
                ptr_to_kernel_arg(&d_outputs.ptr),
                ptr_to_kernel_arg(&d_snapshots_out.ptr),
            ];
            driver.launch(
                kernel,
                batch_size as u32,
                1,
                shared_bytes as u32,
                &mut params,
            )?;
        } else {
            let mut params = [
                ptr_to_kernel_arg(&d_activation.ptr),
                ptr_to_kernel_arg(&d_bias.ptr),
                ptr_to_kernel_arg(&d_response.ptr),
                ptr_to_kernel_arg(&d_output_indices.ptr),
                ptr_to_kernel_arg(&d_source_kind.ptr),
                ptr_to_kernel_arg(&d_source_index.ptr),
                ptr_to_kernel_arg(&d_weights.ptr),
                ptr_to_kernel_arg(&d_incoming_offsets.ptr),
                ptr_to_kernel_arg(&batch_size_i32),
                ptr_to_kernel_arg(&input_count_i32),
                ptr_to_kernel_arg(&node_count_i32),
                ptr_to_kernel_arg(&output_count_i32),
                ptr_to_kernel_arg(&d_inputs.ptr),
                ptr_to_kernel_arg(&d_outputs.ptr),
            ];
            driver.launch(
                kernel,
                batch_size as u32,
                1,
                shared_bytes as u32,
                &mut params,
            )?;
        }

        let raw_outputs = d_outputs.copy_to_vec::<f32>(
            batch_size
                .checked_mul(spec.output_indices.len())
                .ok_or_else(|| "policy output host copy size overflow".to_string())?,
        )?;
        let outputs = reshape_outputs(&raw_outputs, batch_size, spec.output_indices.len());
        let snapshots = if spec.is_recurrent() {
            let raw_snapshots = d_snapshots_out
                .as_ref()
                .ok_or_else(|| "missing recurrent snapshot output buffer".to_string())?
                .copy_to_vec::<f32>(
                    batch_size
                        .checked_mul(spec.node_evals.len())
                        .ok_or_else(|| "policy snapshot host copy size overflow".to_string())?,
                )?;
            Some(reshape_snapshots(spec, &raw_snapshots, batch_size))
        } else {
            None
        };

        Ok(CompiledPolicyResult {
            outputs,
            snapshots,
            backend_used: "cuda_native".to_string(),
        })
    }

    fn ensure_supported(spec: &CompiledPolicySpec) -> Result<(), String> {
        compiled_ptx()?;
        let shared_bytes = policy_shared_bytes(spec.node_evals.len())?;
        if shared_bytes > SHARED_LIMIT_BYTES {
            return Err(format!(
                "policy CUDA kernel requires {shared_bytes} bytes shared memory; limit is {SHARED_LIMIT_BYTES}"
            ));
        }
        for node in &spec.node_evals {
            native_activation_code(&node.activation)?;
            if !node.aggregation.trim().eq_ignore_ascii_case("sum") {
                return Err(format!(
                    "native policy CUDA currently supports only sum aggregation; node {} uses {}",
                    node.node_id, node.aggregation
                ));
            }
        }
        Ok(())
    }

    fn reshape_outputs(raw: &[f32], batch_size: usize, output_count: usize) -> Vec<Vec<f64>> {
        let mut outputs = vec![vec![0.0; output_count]; batch_size];
        for batch in 0..batch_size {
            for output in 0..output_count {
                outputs[batch][output] = raw[(batch * output_count) + output] as f64;
            }
        }
        outputs
    }

    fn reshape_snapshots(
        spec: &CompiledPolicySpec,
        raw: &[f32],
        batch_size: usize,
    ) -> Vec<Option<CompiledPolicySnapshot>> {
        let mut snapshots = Vec::with_capacity(batch_size);
        let node_count = spec.node_evals.len();
        for batch in 0..batch_size {
            let mut node_values = BTreeMap::new();
            for (node_index, node) in spec.node_evals.iter().enumerate() {
                let value = raw[(batch * node_count) + node_index] as f64;
                node_values.insert(node.node_id.to_string(), value);
            }
            snapshots.push(Some(CompiledPolicySnapshot { node_values }));
        }
        snapshots
    }

    fn flatten_inputs_as_f32(input_count: usize, rows: &[Vec<f64>]) -> Result<Vec<f32>, String> {
        let mut flat = Vec::with_capacity(
            rows.len()
                .checked_mul(input_count)
                .ok_or_else(|| "policy input buffer size overflow".to_string())?,
        );
        for row in rows {
            if row.len() != input_count {
                return Err(format!(
                    "policy input row size mismatch: expected {input_count}, got {}",
                    row.len()
                ));
            }
            for value in row {
                flat.push(*value as f32);
            }
        }
        Ok(flat)
    }

    fn flatten_snapshot_as_f32(
        spec: &CompiledPolicySpec,
        request: &CompiledPolicyRequest,
        batch_size: usize,
    ) -> Vec<f32> {
        let mut base = vec![0.0f32; spec.node_evals.len()];
        if let Some(snapshot) = &request.snapshot {
            for (node_index, node) in spec.node_evals.iter().enumerate() {
                if let Some(value) = snapshot.node_values.get(&node.node_id.to_string()) {
                    base[node_index] = *value as f32;
                }
            }
        }
        let mut repeated = Vec::with_capacity(batch_size * spec.node_evals.len());
        for _ in 0..batch_size {
            repeated.extend_from_slice(&base);
        }
        repeated
    }

    fn policy_shared_bytes(node_count: usize) -> Result<usize, String> {
        node_count
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| "policy CUDA shared memory size overflow".to_string())
    }

    fn native_activation_code(value: &str) -> Result<i32, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sigmoid" => Ok(0),
            "tanh" => Ok(1),
            "relu" => Ok(2),
            "identity" | "linear" => Ok(3),
            "clamped" => Ok(4),
            "gauss" => Ok(5),
            "sin" => Ok(6),
            "abs" => Ok(7),
            other => Err(format!(
                "native policy CUDA does not support activation {other:?}"
            )),
        }
    }

    struct PackedPolicyModel {
        activation_codes: Vec<i32>,
        bias: Vec<f32>,
        response: Vec<f32>,
        memory_gate_enabled: Vec<u8>,
        memory_gate_bias: Vec<f32>,
        memory_gate_response: Vec<f32>,
        output_indices: Vec<i32>,
        source_kind: Vec<i32>,
        source_index: Vec<i32>,
        weights: Vec<f32>,
        incoming_offsets: Vec<i32>,
    }

    impl PackedPolicyModel {
        fn from_spec(spec: &CompiledPolicySpec) -> Result<Self, String> {
            let mut activation_codes = Vec::with_capacity(spec.node_evals.len());
            let mut bias = Vec::with_capacity(spec.node_evals.len());
            let mut response = Vec::with_capacity(spec.node_evals.len());
            let mut memory_gate_enabled = Vec::with_capacity(spec.node_evals.len());
            let mut memory_gate_bias = Vec::with_capacity(spec.node_evals.len());
            let mut memory_gate_response = Vec::with_capacity(spec.node_evals.len());
            let mut output_indices = Vec::with_capacity(spec.output_indices.len());
            let mut source_kind = Vec::new();
            let mut source_index = Vec::new();
            let mut weights = Vec::new();
            let mut incoming_offsets = Vec::with_capacity(spec.node_evals.len() + 1);

            for &index in &spec.output_indices {
                output_indices.push(index as i32);
            }
            incoming_offsets.push(0);
            for node in &spec.node_evals {
                activation_codes.push(native_activation_code(&node.activation)?);
                bias.push(node.bias as f32);
                response.push(node.response as f32);
                memory_gate_enabled.push(u8::from(node.memory_gate_enabled));
                memory_gate_bias.push(node.memory_gate_bias as f32);
                memory_gate_response.push(node.memory_gate_response as f32);
                for edge in &node.incoming {
                    let kind = match edge.source_kind.trim().to_ascii_lowercase().as_str() {
                        "input" => 0,
                        "node" => 1,
                        other => {
                            return Err(format!(
                                "unsupported policy source kind {other:?} in native CUDA"
                            ))
                        }
                    };
                    source_kind.push(kind);
                    source_index.push(edge.source_index as i32);
                    weights.push(edge.weight as f32);
                }
                incoming_offsets.push(source_kind.len() as i32);
            }

            Ok(Self {
                activation_codes,
                bias,
                response,
                memory_gate_enabled,
                memory_gate_bias,
                memory_gate_response,
                output_indices,
                source_kind,
                source_index,
                weights,
                incoming_offsets,
            })
        }
    }

    fn compiled_ptx() -> Result<&'static str, String> {
        let cached = CUDA_PTX_CACHE.get_or_init(compile_ptx_to_string);
        match cached {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(message) => Err(message.clone()),
        }
    }

    fn compile_ptx_to_string() -> Result<String, String> {
        let clang =
            find_clang().ok_or_else(|| "clang++ not found for CUDA PTX compilation".to_string())?;
        let temp_dir = env::temp_dir().join("neat_rust_cuda_runtime");
        fs::create_dir_all(&temp_dir).map_err(|err| err.to_string())?;
        let src_path = temp_dir.join("gpu_kernels.cu");
        let ptx_path = temp_dir.join("gpu_kernels.ptx");
        fs::write(&src_path, CUDA_KERNEL_SOURCE).map_err(|err| err.to_string())?;
        let output = Command::new(&clang)
            .arg("--cuda-device-only")
            .arg("--cuda-gpu-arch=sm_50")
            .arg("-nocudainc")
            .arg("-nocudalib")
            .arg("-O2")
            .arg("-S")
            .arg(&src_path)
            .arg("-o")
            .arg(&ptx_path)
            .output()
            .map_err(|err| format!("failed to run {}: {err}", clang.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "failed to compile CUDA PTX with {}: {}",
                clang.display(),
                stderr.trim()
            ));
        }
        let ptx = fs::read_to_string(&ptx_path).map_err(|err| err.to_string())?;
        if ptx.trim().is_empty() {
            Err("compiled CUDA PTX was empty".to_string())
        } else {
            Ok(ptx)
        }
    }

    fn find_clang() -> Option<PathBuf> {
        if let Ok(path) = env::var("CLANG_CUDA") {
            let candidate = PathBuf::from(path);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bundled = manifest_dir.parent().map(|root| {
            root.join(".tools")
                .join("llvm-mingw")
                .join("bin")
                .join("clang++.exe")
        });
        if let Some(candidate) = bundled {
            if candidate.exists() {
                return Some(candidate);
            }
        }

        let output = Command::new("where").arg("clang++").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let first = stdout
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?;
        Some(PathBuf::from(first))
    }

    fn ptr_to_kernel_arg<T>(value: &T) -> *mut c_void {
        value as *const T as *mut c_void
    }

    struct CudaDriver {
        _library: LibraryHandle,
        cu_init: CuInit,
        cu_device_get_count: CuDeviceGetCount,
        cu_device_get: CuDeviceGet,
        cu_ctx_create: CuCtxCreate,
        cu_ctx_destroy: CuCtxDestroy,
        cu_module_load_data_ex: CuModuleLoadDataEx,
        cu_module_unload: CuModuleUnload,
        cu_module_get_function: CuModuleGetFunction,
        cu_mem_alloc: CuMemAlloc,
        cu_mem_free: CuMemFree,
        cu_memcpy_htod: CuMemcpyHtoD,
        cu_memcpy_dtoh: CuMemcpyDtoH,
        cu_launch_kernel: CuLaunchKernel,
        cu_ctx_synchronize: CuCtxSynchronize,
        device: CuDevice,
    }

    impl CudaDriver {
        fn load() -> Result<Self, String> {
            let library = LibraryHandle::open("nvcuda.dll")
                .ok_or_else(|| "nvcuda.dll is unavailable".to_string())?;
            let cu_init = library
                .symbol(b"cuInit\0")
                .ok_or_else(|| "cuInit unavailable".to_string())?;
            let cu_device_get_count = library
                .symbol(b"cuDeviceGetCount\0")
                .ok_or_else(|| "cuDeviceGetCount unavailable".to_string())?;
            let cu_device_get = library
                .symbol(b"cuDeviceGet\0")
                .ok_or_else(|| "cuDeviceGet unavailable".to_string())?;
            let cu_ctx_create = library
                .symbol(b"cuCtxCreate_v2\0")
                .ok_or_else(|| "cuCtxCreate_v2 unavailable".to_string())?;
            let cu_ctx_destroy = library
                .symbol(b"cuCtxDestroy_v2\0")
                .ok_or_else(|| "cuCtxDestroy_v2 unavailable".to_string())?;
            let cu_module_load_data_ex = library
                .symbol(b"cuModuleLoadDataEx\0")
                .ok_or_else(|| "cuModuleLoadDataEx unavailable".to_string())?;
            let cu_module_unload = library
                .symbol(b"cuModuleUnload\0")
                .ok_or_else(|| "cuModuleUnload unavailable".to_string())?;
            let cu_module_get_function = library
                .symbol(b"cuModuleGetFunction\0")
                .ok_or_else(|| "cuModuleGetFunction unavailable".to_string())?;
            let cu_mem_alloc = library
                .symbol(b"cuMemAlloc_v2\0")
                .ok_or_else(|| "cuMemAlloc_v2 unavailable".to_string())?;
            let cu_mem_free = library
                .symbol(b"cuMemFree_v2\0")
                .ok_or_else(|| "cuMemFree_v2 unavailable".to_string())?;
            let cu_memcpy_htod = library
                .symbol(b"cuMemcpyHtoD_v2\0")
                .ok_or_else(|| "cuMemcpyHtoD_v2 unavailable".to_string())?;
            let cu_memcpy_dtoh = library
                .symbol(b"cuMemcpyDtoH_v2\0")
                .ok_or_else(|| "cuMemcpyDtoH_v2 unavailable".to_string())?;
            let cu_launch_kernel = library
                .symbol(b"cuLaunchKernel\0")
                .ok_or_else(|| "cuLaunchKernel unavailable".to_string())?;
            let cu_ctx_synchronize = library
                .symbol(b"cuCtxSynchronize\0")
                .ok_or_else(|| "cuCtxSynchronize unavailable".to_string())?;

            let driver = Self {
                _library: library,
                cu_init,
                cu_device_get_count,
                cu_device_get,
                cu_ctx_create,
                cu_ctx_destroy,
                cu_module_load_data_ex,
                cu_module_unload,
                cu_module_get_function,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_launch_kernel,
                cu_ctx_synchronize,
                device: 0,
            };
            driver.unavailable_result(unsafe { (driver.cu_init)(0) }, "cuInit")?;

            let mut count = 0;
            driver.unavailable_result(
                unsafe { (driver.cu_device_get_count)(&mut count) },
                "cuDeviceGetCount",
            )?;
            if count < 1 {
                return Err("no CUDA device found".to_string());
            }

            let mut device = 0;
            driver.unavailable_result(
                unsafe { (driver.cu_device_get)(&mut device, 0) },
                "cuDeviceGet",
            )?;
            Ok(Self { device, ..driver })
        }

        fn create_context(&self) -> Result<CudaContext<'_>, String> {
            let mut context = ptr::null_mut();
            self.result(
                unsafe { (self.cu_ctx_create)(&mut context, 0, self.device) },
                "cuCtxCreate_v2",
            )?;
            Ok(CudaContext {
                driver: self,
                context,
            })
        }

        fn result(&self, code: CuResult, label: &str) -> Result<(), String> {
            if code == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!("{label} failed with CUDA error {code}"))
            }
        }

        fn unavailable_result(&self, code: CuResult, label: &str) -> Result<(), String> {
            if code == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!("{label} is unavailable"))
            }
        }

        fn launch(
            &self,
            function: CuFunction,
            grid_x: u32,
            block_x: u32,
            shared_mem_bytes: u32,
            params: &mut [*mut c_void],
        ) -> Result<(), String> {
            self.result(
                unsafe {
                    (self.cu_launch_kernel)(
                        function,
                        grid_x,
                        1,
                        1,
                        block_x,
                        1,
                        1,
                        shared_mem_bytes,
                        ptr::null_mut(),
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel",
            )?;
            self.result(unsafe { (self.cu_ctx_synchronize)() }, "cuCtxSynchronize")
        }
    }

    struct CudaContext<'a> {
        driver: &'a CudaDriver,
        context: CuContext,
    }

    impl<'a> CudaContext<'a> {
        fn load_module(&self, ptx: &str) -> Result<CudaModule<'a>, String> {
            let ptx = CString::new(ptx).map_err(|_| "PTX contains NUL byte".to_string())?;
            let mut module = ptr::null_mut();
            self.driver.result(
                unsafe {
                    (self.driver.cu_module_load_data_ex)(
                        &mut module,
                        ptx.as_ptr() as *const c_void,
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                },
                "cuModuleLoadDataEx",
            )?;
            Ok(CudaModule {
                driver: self.driver,
                module,
            })
        }
    }

    impl Drop for CudaContext<'_> {
        fn drop(&mut self) {
            if !self.context.is_null() {
                unsafe {
                    (self.driver.cu_ctx_destroy)(self.context);
                }
            }
        }
    }

    struct CudaModule<'a> {
        driver: &'a CudaDriver,
        module: CuModule,
    }

    impl<'a> CudaModule<'a> {
        fn function(&self, name: &[u8]) -> Result<CuFunction, String> {
            let mut function = ptr::null_mut();
            self.driver.result(
                unsafe {
                    (self.driver.cu_module_get_function)(
                        &mut function,
                        self.module,
                        name.as_ptr() as *const c_char,
                    )
                },
                "cuModuleGetFunction",
            )?;
            Ok(function)
        }
    }

    impl Drop for CudaModule<'_> {
        fn drop(&mut self) {
            if !self.module.is_null() {
                unsafe {
                    (self.driver.cu_module_unload)(self.module);
                }
            }
        }
    }

    struct DeviceBuffer<'a> {
        driver: &'a CudaDriver,
        ptr: CuDevicePtr,
    }

    impl<'a> DeviceBuffer<'a> {
        fn from_slice<T: Copy>(driver: &'a CudaDriver, values: &[T]) -> Result<Self, String> {
            let bytes = mem::size_of_val(values);
            let mut ptr = 0;
            driver.result(
                unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) },
                "cuMemAlloc_v2",
            )?;
            if bytes > 0 {
                driver.result(
                    unsafe {
                        (driver.cu_memcpy_htod)(ptr, values.as_ptr() as *const c_void, bytes)
                    },
                    "cuMemcpyHtoD_v2",
                )?;
            }
            Ok(Self { driver, ptr })
        }

        fn zeroed<T>(driver: &'a CudaDriver, len: usize) -> Result<Self, String> {
            let bytes = len
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| "native CUDA device allocation size overflow".to_string())?;
            let mut ptr = 0;
            driver.result(
                unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) },
                "cuMemAlloc_v2",
            )?;
            if bytes > 0 {
                let zeros = vec![0u8; bytes];
                driver.result(
                    unsafe { (driver.cu_memcpy_htod)(ptr, zeros.as_ptr() as *const c_void, bytes) },
                    "cuMemcpyHtoD_v2",
                )?;
            }
            Ok(Self { driver, ptr })
        }

        fn copy_to_vec<T: Copy + Default>(&self, len: usize) -> Result<Vec<T>, String> {
            let bytes = len
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| "native CUDA host copy size overflow".to_string())?;
            let mut values = vec![T::default(); len];
            if bytes > 0 {
                self.driver.result(
                    unsafe {
                        (self.driver.cu_memcpy_dtoh)(
                            values.as_mut_ptr() as *mut c_void,
                            self.ptr,
                            bytes,
                        )
                    },
                    "cuMemcpyDtoH_v2",
                )?;
            }
            Ok(values)
        }
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
            if self.ptr != 0 {
                unsafe {
                    (self.driver.cu_mem_free)(self.ptr);
                }
            }
        }
    }

    struct LibraryHandle {
        handle: *mut c_void,
    }

    impl LibraryHandle {
        fn open(name: &str) -> Option<Self> {
            let wide: Vec<u16> = OsStr::new(name)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle.is_null() {
                None
            } else {
                Some(Self { handle })
            }
        }

        fn symbol<T: Copy>(&self, name: &[u8]) -> Option<T> {
            let ptr = unsafe { GetProcAddress(self.handle, name.as_ptr() as *const c_char) };
            if ptr.is_null() {
                None
            } else {
                Some(unsafe { mem::transmute_copy(&ptr) })
            }
        }
    }

    impl Drop for LibraryHandle {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    FreeLibrary(self.handle);
                }
            }
        }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }
}
