#[cfg(windows)]
use crate::activation::ActivationFunction;
use crate::gpu::{
    GpuEvaluatorError, GpuInputBatch, OutputTrajectory, PackedCTRNNPopulation, PackedIZNNPopulation,
};

#[cfg(windows)]
const CUDA_KERNEL_SOURCE: &str = include_str!("../cuda_kernels.cu");

pub(crate) fn native_cuda_available() -> bool {
    imp::native_cuda_available()
}

pub(crate) fn ctrnn_native_supported(
    packed: &PackedCTRNNPopulation,
) -> Result<(), GpuEvaluatorError> {
    imp::ctrnn_native_supported(packed)
}

pub(crate) fn iznn_native_supported(
    packed: &PackedIZNNPopulation,
) -> Result<(), GpuEvaluatorError> {
    imp::iznn_native_supported(packed)
}

pub(crate) fn evaluate_ctrnn_batch_native<I>(
    packed: &PackedCTRNNPopulation,
    dt: f64,
    t_max: f64,
    input_fn: &mut I,
) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
{
    imp::evaluate_ctrnn_batch_native(packed, dt, t_max, input_fn)
}

pub(crate) fn evaluate_iznn_batch_native<I>(
    packed: &PackedIZNNPopulation,
    dt: f64,
    t_max: f64,
    input_fn: &mut I,
) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
{
    imp::evaluate_iznn_batch_native(packed, dt, t_max, input_fn)
}

#[cfg(not(windows))]
mod imp {
    use super::*;

    pub(crate) fn native_cuda_available() -> bool {
        false
    }

    pub(crate) fn ctrnn_native_supported(
        _packed: &PackedCTRNNPopulation,
    ) -> Result<(), GpuEvaluatorError> {
        Err(GpuEvaluatorError::NativeBackendUnavailable)
    }

    pub(crate) fn iznn_native_supported(
        _packed: &PackedIZNNPopulation,
    ) -> Result<(), GpuEvaluatorError> {
        Err(GpuEvaluatorError::NativeBackendUnavailable)
    }

    pub(crate) fn evaluate_ctrnn_batch_native<I>(
        _packed: &PackedCTRNNPopulation,
        _dt: f64,
        _t_max: f64,
        _input_fn: &mut I,
    ) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
    where
        I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    {
        Err(GpuEvaluatorError::NativeBackendUnavailable)
    }

    pub(crate) fn evaluate_iznn_batch_native<I>(
        _packed: &PackedIZNNPopulation,
        _dt: f64,
        _t_max: f64,
        _input_fn: &mut I,
    ) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
    where
        I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    {
        Err(GpuEvaluatorError::NativeBackendUnavailable)
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
    const CTRNN_SHARED_MULTIPLIER: usize = 2;
    const IZNN_SHARED_MULTIPLIER: usize = 4;

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
        if !ptx_available() {
            return false;
        }
        CudaDriver::load().is_ok()
    }

    pub(crate) fn ctrnn_native_supported(
        packed: &PackedCTRNNPopulation,
    ) -> Result<(), GpuEvaluatorError> {
        ensure_ptx_available()?;
        ensure_shared_memory(packed.max_nodes, CTRNN_SHARED_MULTIPLIER)?;
        map_ctrnn_activation_codes(packed).map(|_| ())
    }

    pub(crate) fn iznn_native_supported(
        packed: &PackedIZNNPopulation,
    ) -> Result<(), GpuEvaluatorError> {
        ensure_ptx_available()?;
        ensure_shared_memory(packed.max_nodes, IZNN_SHARED_MULTIPLIER)
    }

    pub(crate) fn evaluate_ctrnn_batch_native<I>(
        packed: &PackedCTRNNPopulation,
        dt: f64,
        t_max: f64,
        input_fn: &mut I,
    ) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
    where
        I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    {
        ensure_ptx_available()?;
        ensure_shared_memory(packed.max_nodes, CTRNN_SHARED_MULTIPLIER)?;

        let population_size = packed.genome_keys.len();
        let num_steps = (t_max / dt) as usize;
        if num_steps == 0 {
            return Ok(vec![Vec::new(); population_size]);
        }

        let driver = CudaDriver::load()?;
        let context = driver.create_context()?;
        let module = context.load_module(compiled_ptx()?)?;
        let kernel = module.function(b"ctrnn_eval_kernel\0")?;

        let activation_codes = map_ctrnn_activation_codes(packed)?;
        let inputs =
            collect_inputs_as_f32(population_size, num_steps, packed.num_inputs, dt, input_fn)?;
        let output_len = population_size * num_steps * packed.num_outputs;

        let weights = to_f32_vec(&packed.weights);
        let bias = to_f32_vec(&packed.bias);
        let response = to_f32_vec(&packed.response);
        let tau = to_f32_vec(&packed.tau);
        let node_mask = bools_to_u8(&packed.node_mask);

        let d_weights = DeviceBuffer::from_slice(&driver, &weights)?;
        let d_bias = DeviceBuffer::from_slice(&driver, &bias)?;
        let d_response = DeviceBuffer::from_slice(&driver, &response)?;
        let d_tau = DeviceBuffer::from_slice(&driver, &tau)?;
        let d_activation = DeviceBuffer::from_slice(&driver, &activation_codes)?;
        let d_mask = DeviceBuffer::from_slice(&driver, &node_mask)?;
        let d_inputs = DeviceBuffer::from_slice(&driver, &inputs)?;
        let d_outputs = DeviceBuffer::zeroed::<f32>(&driver, output_len)?;

        let population_size_i32 = population_size as i32;
        let max_nodes_i32 = packed.max_nodes as i32;
        let num_inputs_i32 = packed.num_inputs as i32;
        let num_outputs_i32 = packed.num_outputs as i32;
        let num_steps_i32 = num_steps as i32;
        let dt_f32 = dt as f32;

        let mut params = [
            ptr_to_kernel_arg(&d_weights.ptr),
            ptr_to_kernel_arg(&d_bias.ptr),
            ptr_to_kernel_arg(&d_response.ptr),
            ptr_to_kernel_arg(&d_tau.ptr),
            ptr_to_kernel_arg(&d_activation.ptr),
            ptr_to_kernel_arg(&d_mask.ptr),
            ptr_to_kernel_arg(&population_size_i32),
            ptr_to_kernel_arg(&max_nodes_i32),
            ptr_to_kernel_arg(&num_inputs_i32),
            ptr_to_kernel_arg(&num_outputs_i32),
            ptr_to_kernel_arg(&num_steps_i32),
            ptr_to_kernel_arg(&dt_f32),
            ptr_to_kernel_arg(&d_inputs.ptr),
            ptr_to_kernel_arg(&d_outputs.ptr),
        ];

        driver.launch(
            kernel,
            population_size as u32,
            1,
            ctrnn_shared_bytes(packed.max_nodes)? as u32,
            &mut params,
        )?;
        let outputs = d_outputs.copy_to_vec::<f32>(output_len)?;
        Ok(outputs_to_trajectories(
            &outputs,
            population_size,
            num_steps,
            packed.num_outputs,
        ))
    }

    pub(crate) fn evaluate_iznn_batch_native<I>(
        packed: &PackedIZNNPopulation,
        dt: f64,
        t_max: f64,
        input_fn: &mut I,
    ) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
    where
        I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    {
        ensure_ptx_available()?;
        ensure_shared_memory(packed.max_nodes, IZNN_SHARED_MULTIPLIER)?;

        let population_size = packed.genome_keys.len();
        let num_steps = (t_max / dt) as usize;
        if num_steps == 0 {
            return Ok(vec![Vec::new(); population_size]);
        }

        let driver = CudaDriver::load()?;
        let context = driver.create_context()?;
        let module = context.load_module(compiled_ptx()?)?;
        let kernel = module.function(b"iznn_eval_kernel\0")?;

        let inputs =
            collect_inputs_as_f32(population_size, num_steps, packed.num_inputs, dt, input_fn)?;
        let output_len = population_size * num_steps * packed.num_outputs;

        let weights = to_f32_vec(&packed.weights);
        let bias = to_f32_vec(&packed.bias);
        let a = to_f32_vec(&packed.a);
        let b = to_f32_vec(&packed.b);
        let c = to_f32_vec(&packed.c);
        let d = to_f32_vec(&packed.d);
        let node_mask = bools_to_u8(&packed.node_mask);

        let d_weights = DeviceBuffer::from_slice(&driver, &weights)?;
        let d_bias = DeviceBuffer::from_slice(&driver, &bias)?;
        let d_a = DeviceBuffer::from_slice(&driver, &a)?;
        let d_b = DeviceBuffer::from_slice(&driver, &b)?;
        let d_c = DeviceBuffer::from_slice(&driver, &c)?;
        let d_d = DeviceBuffer::from_slice(&driver, &d)?;
        let d_mask = DeviceBuffer::from_slice(&driver, &node_mask)?;
        let d_inputs = DeviceBuffer::from_slice(&driver, &inputs)?;
        let d_outputs = DeviceBuffer::zeroed::<f32>(&driver, output_len)?;

        let population_size_i32 = population_size as i32;
        let max_nodes_i32 = packed.max_nodes as i32;
        let num_inputs_i32 = packed.num_inputs as i32;
        let num_outputs_i32 = packed.num_outputs as i32;
        let num_steps_i32 = num_steps as i32;
        let dt_f32 = dt as f32;

        let mut params = [
            ptr_to_kernel_arg(&d_weights.ptr),
            ptr_to_kernel_arg(&d_bias.ptr),
            ptr_to_kernel_arg(&d_a.ptr),
            ptr_to_kernel_arg(&d_b.ptr),
            ptr_to_kernel_arg(&d_c.ptr),
            ptr_to_kernel_arg(&d_d.ptr),
            ptr_to_kernel_arg(&d_mask.ptr),
            ptr_to_kernel_arg(&population_size_i32),
            ptr_to_kernel_arg(&max_nodes_i32),
            ptr_to_kernel_arg(&num_inputs_i32),
            ptr_to_kernel_arg(&num_outputs_i32),
            ptr_to_kernel_arg(&num_steps_i32),
            ptr_to_kernel_arg(&dt_f32),
            ptr_to_kernel_arg(&d_inputs.ptr),
            ptr_to_kernel_arg(&d_outputs.ptr),
        ];

        driver.launch(
            kernel,
            population_size as u32,
            1,
            iznn_shared_bytes(packed.max_nodes)? as u32,
            &mut params,
        )?;
        let outputs = d_outputs.copy_to_vec::<f32>(output_len)?;
        Ok(outputs_to_trajectories(
            &outputs,
            population_size,
            num_steps,
            packed.num_outputs,
        ))
    }

    fn ensure_ptx_available() -> Result<(), GpuEvaluatorError> {
        compiled_ptx().map(|_| ())
    }

    fn ptx_available() -> bool {
        compiled_ptx().is_ok()
    }

    fn compiled_ptx() -> Result<&'static str, GpuEvaluatorError> {
        let cached = CUDA_PTX_CACHE.get_or_init(compile_ptx_to_string);
        match cached {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(_message) => Err(GpuEvaluatorError::NativeBackendUnavailable),
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

    fn collect_inputs_as_f32<I>(
        population_size: usize,
        num_steps: usize,
        num_inputs: usize,
        dt: f64,
        input_fn: &mut I,
    ) -> Result<Vec<f32>, GpuEvaluatorError>
    where
        I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    {
        let mut result = Vec::with_capacity(population_size * num_steps * num_inputs);
        for step in 0..num_steps {
            let time = step as f64 * dt;
            let inputs = input_fn(time, dt, population_size, num_inputs)
                .expand(population_size, num_inputs)?;
            for row in inputs {
                for value in row {
                    result.push(value as f32);
                }
            }
        }
        Ok(result)
    }

    fn outputs_to_trajectories(
        outputs: &[f32],
        population_size: usize,
        num_steps: usize,
        num_outputs: usize,
    ) -> Vec<OutputTrajectory> {
        let mut trajectories = vec![vec![vec![0.0; num_outputs]; num_steps]; population_size];
        for (genome_idx, trajectory) in trajectories.iter_mut().enumerate().take(population_size) {
            for (step, step_outputs) in trajectory.iter_mut().enumerate().take(num_steps) {
                let base = (genome_idx * num_steps + step) * num_outputs;
                for output_idx in 0..num_outputs {
                    step_outputs[output_idx] = outputs[base + output_idx] as f64;
                }
            }
        }
        trajectories
    }

    fn ensure_shared_memory(max_nodes: usize, multiplier: usize) -> Result<(), GpuEvaluatorError> {
        let bytes = max_nodes
            .checked_mul(multiplier)
            .and_then(|value| value.checked_mul(mem::size_of::<f32>()))
            .ok_or_else(|| {
                GpuEvaluatorError::NativeUnsupported(
                    "native CUDA shared memory size overflow".to_string(),
                )
            })?;
        if bytes > SHARED_LIMIT_BYTES {
            return Err(GpuEvaluatorError::NativeUnsupported(format!(
                "native CUDA kernel requires {bytes} bytes shared memory; limit is {SHARED_LIMIT_BYTES}"
            )));
        }
        Ok(())
    }

    fn ctrnn_shared_bytes(max_nodes: usize) -> Result<usize, GpuEvaluatorError> {
        max_nodes
            .checked_mul(CTRNN_SHARED_MULTIPLIER)
            .and_then(|value| value.checked_mul(mem::size_of::<f32>()))
            .ok_or_else(|| {
                GpuEvaluatorError::NativeUnsupported(
                    "native CUDA shared memory size overflow".to_string(),
                )
            })
    }

    fn iznn_shared_bytes(max_nodes: usize) -> Result<usize, GpuEvaluatorError> {
        max_nodes
            .checked_mul(IZNN_SHARED_MULTIPLIER)
            .and_then(|value| value.checked_mul(mem::size_of::<f32>()))
            .ok_or_else(|| {
                GpuEvaluatorError::NativeUnsupported(
                    "native CUDA shared memory size overflow".to_string(),
                )
            })
    }

    fn map_ctrnn_activation_codes(
        packed: &PackedCTRNNPopulation,
    ) -> Result<Vec<i32>, GpuEvaluatorError> {
        let mut codes = vec![3i32; packed.activation.len()];
        for genome_idx in 0..packed.genome_keys.len() {
            for dense_idx in 0..packed.max_nodes {
                let flat_idx = genome_idx * packed.max_nodes + dense_idx;
                if !packed.node_mask[flat_idx] || dense_idx < packed.num_inputs {
                    continue;
                }
                codes[flat_idx] =
                    native_activation_code(packed.activation[flat_idx]).ok_or_else(|| {
                        GpuEvaluatorError::UnsupportedActivation {
                            genome_key: packed.genome_keys[genome_idx],
                            node_key: dense_key(&packed.node_key_maps[genome_idx], dense_idx),
                            name: packed.activation[flat_idx].name().to_string(),
                        }
                    })?;
            }
        }
        Ok(codes)
    }

    fn native_activation_code(value: ActivationFunction) -> Option<i32> {
        match value {
            ActivationFunction::Sigmoid => Some(0),
            ActivationFunction::Tanh => Some(1),
            ActivationFunction::Relu => Some(2),
            ActivationFunction::Identity => Some(3),
            ActivationFunction::Clamped => Some(4),
            ActivationFunction::Elu => Some(5),
            ActivationFunction::Gauss => Some(6),
            ActivationFunction::Abs => Some(7),
            ActivationFunction::Square => Some(8),
            _ => None,
        }
    }

    fn dense_key(key_map: &std::collections::BTreeMap<i64, usize>, dense_idx: usize) -> i64 {
        key_map
            .iter()
            .find(|(_, idx)| **idx == dense_idx)
            .map(|(key, _)| *key)
            .unwrap_or(dense_idx as i64)
    }

    fn bools_to_u8(values: &[bool]) -> Vec<u8> {
        values.iter().map(|value| u8::from(*value)).collect()
    }

    fn to_f32_vec(values: &[f64]) -> Vec<f32> {
        values.iter().map(|value| *value as f32).collect()
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
        fn load() -> Result<Self, GpuEvaluatorError> {
            let library = LibraryHandle::open("nvcuda.dll")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_init = library
                .symbol(b"cuInit\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_device_get_count = library
                .symbol(b"cuDeviceGetCount\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_device_get = library
                .symbol(b"cuDeviceGet\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_ctx_create = library
                .symbol(b"cuCtxCreate_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_ctx_destroy = library
                .symbol(b"cuCtxDestroy_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_module_load_data_ex = library
                .symbol(b"cuModuleLoadDataEx\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_module_unload = library
                .symbol(b"cuModuleUnload\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_module_get_function = library
                .symbol(b"cuModuleGetFunction\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_mem_alloc = library
                .symbol(b"cuMemAlloc_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_mem_free = library
                .symbol(b"cuMemFree_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_memcpy_htod = library
                .symbol(b"cuMemcpyHtoD_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_memcpy_dtoh = library
                .symbol(b"cuMemcpyDtoH_v2\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_launch_kernel = library
                .symbol(b"cuLaunchKernel\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;
            let cu_ctx_synchronize = library
                .symbol(b"cuCtxSynchronize\0")
                .ok_or(GpuEvaluatorError::NativeBackendUnavailable)?;

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
                return Err(GpuEvaluatorError::NativeBackendUnavailable);
            }

            let mut device = 0;
            driver.unavailable_result(
                unsafe { (driver.cu_device_get)(&mut device, 0) },
                "cuDeviceGet",
            )?;

            Ok(Self { device, ..driver })
        }

        fn create_context(&self) -> Result<CudaContext<'_>, GpuEvaluatorError> {
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

        fn result(&self, code: CuResult, label: &str) -> Result<(), GpuEvaluatorError> {
            if code == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(GpuEvaluatorError::NativeDriver(format!(
                    "{label} failed with CUDA error {code}"
                )))
            }
        }

        fn unavailable_result(
            &self,
            code: CuResult,
            _label: &str,
        ) -> Result<(), GpuEvaluatorError> {
            if code == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(GpuEvaluatorError::NativeBackendUnavailable)
            }
        }

        fn launch(
            &self,
            function: CuFunction,
            grid_x: u32,
            block_x: u32,
            shared_mem_bytes: u32,
            params: &mut [*mut c_void],
        ) -> Result<(), GpuEvaluatorError> {
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
        fn load_module(&self, ptx: &str) -> Result<CudaModule<'a>, GpuEvaluatorError> {
            let ptx = CString::new(ptx).map_err(|_| {
                GpuEvaluatorError::NativeDriver("PTX contains NUL byte".to_string())
            })?;
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
        fn function(&self, name: &[u8]) -> Result<CuFunction, GpuEvaluatorError> {
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
        fn from_slice<T: Copy>(
            driver: &'a CudaDriver,
            values: &[T],
        ) -> Result<Self, GpuEvaluatorError> {
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

        fn zeroed<T>(driver: &'a CudaDriver, len: usize) -> Result<Self, GpuEvaluatorError> {
            let bytes = len.checked_mul(mem::size_of::<T>()).ok_or_else(|| {
                GpuEvaluatorError::NativeUnsupported(
                    "native CUDA device allocation size overflow".to_string(),
                )
            })?;
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

        fn copy_to_vec<T: Copy + Default>(&self, len: usize) -> Result<Vec<T>, GpuEvaluatorError> {
            let bytes = len.checked_mul(mem::size_of::<T>()).ok_or_else(|| {
                GpuEvaluatorError::NativeUnsupported(
                    "native CUDA host copy size overflow".to_string(),
                )
            })?;
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
