#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(t, b) __attribute__((launch_bounds(t, b)))
#define __device_builtin__ __attribute__((device_builtin))
#define __cudart_builtin__ __attribute__((cudart_builtin))

__device__ inline unsigned int tid_x() {
  unsigned int x;
  asm("mov.u32 %0, %%tid.x;" : "=r"(x));
  return x;
}

__device__ inline unsigned int ctaid_x() {
  unsigned int x;
  asm("mov.u32 %0, %%ctaid.x;" : "=r"(x));
  return x;
}

__device__ inline float absf(float x) {
  return x < 0.0f ? -x : x;
}

__device__ inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

__device__ inline float fast_exp(float x) {
  x = clampf(x, -60.0f, 60.0f);
  float y = 1.0f + (x * 0.0009765625f);
  for (int i = 0; i < 10; ++i) {
    y *= y;
  }
  return y;
}

__device__ inline float fast_tanh(float x) {
  x = clampf(x, -10.0f, 10.0f);
  float ax = absf(x);
  if (ax < 1.0f) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + (9.0f * x2));
  }
  float e = fast_exp(2.0f * x);
  return (e - 1.0f) / (e + 1.0f);
}

__device__ inline float apply_activation(int code, float x) {
  switch (code) {
    case 0: {
      float z = clampf(5.0f * x, -60.0f, 60.0f);
      return 1.0f / (1.0f + fast_exp(-z));
    }
    case 1:
      return fast_tanh(2.5f * x);
    case 2:
      return x > 0.0f ? x : 0.0f;
    case 3:
      return x;
    case 4:
      return clampf(x, -1.0f, 1.0f);
    case 5:
      return x > 0.0f ? x : (fast_exp(x) - 1.0f);
    case 6: {
      float z = clampf(x, -3.4f, 3.4f);
      return fast_exp(-5.0f * z * z);
    }
    case 7:
      return absf(x);
    case 8:
      return x * x;
    default:
      return x;
  }
}

extern "C" __global__ void ctrnn_eval_kernel(
    const float* weights,
    const float* bias,
    const float* response,
    const float* tau,
    const int* activation,
    const unsigned char* node_mask,
    int population_size,
    int max_nodes,
    int num_inputs,
    int num_outputs,
    int num_steps,
    float dt,
    const float* inputs,
    float* outputs) {
  if (tid_x() != 0) {
    return;
  }

  int genome_idx = (int)ctaid_x();
  if (genome_idx >= population_size) {
    return;
  }

  extern __shared__ float shared[];
  float* state = shared;
  float* next_state = shared + max_nodes;

  for (int idx = 0; idx < max_nodes; ++idx) {
    state[idx] = 0.0f;
    next_state[idx] = 0.0f;
  }

  int node_base = genome_idx * max_nodes;
  int input_base = genome_idx * num_steps * num_inputs;
  int output_base = genome_idx * num_steps * num_outputs;

  for (int step = 0; step < num_steps; ++step) {
    int step_input = input_base + (step * num_inputs);
    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
      state[input_idx] = inputs[step_input + input_idx];
    }

    for (int dst_idx = 0; dst_idx < max_nodes; ++dst_idx) {
      int flat_idx = node_base + dst_idx;
      if (!node_mask[flat_idx]) {
        next_state[dst_idx] = 0.0f;
        continue;
      }

      if (dst_idx < num_inputs) {
        next_state[dst_idx] = inputs[step_input + dst_idx];
        continue;
      }

      float weighted_sum = 0.0f;
      int weight_base = (node_base + dst_idx) * max_nodes;
      for (int src_idx = 0; src_idx < max_nodes; ++src_idx) {
        weighted_sum += weights[weight_base + src_idx] * state[src_idx];
      }

      float tau_value = tau[flat_idx];
      if (tau_value < 1.0e-6f) {
        tau_value = 1.0e-6f;
      }
      float activation_input = bias[flat_idx] + (response[flat_idx] * weighted_sum);
      float z = apply_activation(activation[flat_idx], activation_input);
      float decay = fast_exp(-dt / tau_value);
      next_state[dst_idx] = (decay * state[dst_idx]) + ((1.0f - decay) * z);
    }

    float* tmp = state;
    state = next_state;
    next_state = tmp;

    int step_output = output_base + (step * num_outputs);
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      outputs[step_output + output_idx] = state[num_inputs + output_idx];
    }
  }
}

extern "C" __global__ void iznn_eval_kernel(
    const float* weights,
    const float* bias,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    const unsigned char* node_mask,
    int population_size,
    int max_nodes,
    int num_inputs,
    int num_outputs,
    int num_steps,
    float dt,
    const float* inputs,
    float* outputs) {
  if (tid_x() != 0) {
    return;
  }

  int genome_idx = (int)ctaid_x();
  if (genome_idx >= population_size) {
    return;
  }

  extern __shared__ float shared[];
  float* v = shared;
  float* u = shared + max_nodes;
  float* fired = shared + (2 * max_nodes);
  float* source = shared + (3 * max_nodes);

  int node_base = genome_idx * max_nodes;
  int input_base = genome_idx * num_steps * num_inputs;
  int output_base = genome_idx * num_steps * num_outputs;

  for (int idx = 0; idx < max_nodes; ++idx) {
    int flat_idx = node_base + idx;
    if (node_mask[flat_idx]) {
      v[idx] = c[flat_idx];
      u[idx] = b[flat_idx] * v[idx];
    } else {
      v[idx] = 0.0f;
      u[idx] = 0.0f;
    }
    fired[idx] = 0.0f;
    source[idx] = 0.0f;
  }

  for (int step = 0; step < num_steps; ++step) {
    int step_input = input_base + (step * num_inputs);
    for (int idx = 0; idx < max_nodes; ++idx) {
      source[idx] = fired[idx];
      fired[idx] = 0.0f;
    }
    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
      source[input_idx] = inputs[step_input + input_idx];
    }

    for (int dst_idx = num_inputs; dst_idx < max_nodes; ++dst_idx) {
      int flat_idx = node_base + dst_idx;
      if (!node_mask[flat_idx]) {
        fired[dst_idx] = 0.0f;
        continue;
      }

      float current = bias[flat_idx];
      int weight_base = (node_base + dst_idx) * max_nodes;
      for (int src_idx = 0; src_idx < max_nodes; ++src_idx) {
        current += weights[weight_base + src_idx] * source[src_idx];
      }

      float vv = v[dst_idx];
      float uu = u[dst_idx];
      vv += dt * ((0.04f * vv * vv) + (5.0f * vv) + 140.0f - uu + current);
      uu += dt * (a[flat_idx] * ((b[flat_idx] * vv) - uu));

      if (vv > 30.0f) {
        fired[dst_idx] = 1.0f;
        v[dst_idx] = c[flat_idx];
        u[dst_idx] = uu + d[flat_idx];
      } else {
        fired[dst_idx] = 0.0f;
        v[dst_idx] = vv;
        u[dst_idx] = uu;
      }
    }

    int step_output = output_base + (step * num_outputs);
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      outputs[step_output + output_idx] = fired[num_inputs + output_idx];
    }
  }
}

__device__ inline float policy_sigmoid(float x) {
  return 1.0f / (1.0f + fast_exp(-clampf(x, -60.0f, 60.0f)));
}

__device__ inline float policy_gauss(float x) {
  return fast_exp(-(x * x));
}

__device__ inline float policy_sin(float x) {
  x = clampf(x, -6.2831853f, 6.2831853f);
  float x2 = x * x;
  float x3 = x2 * x;
  float x5 = x3 * x2;
  float x7 = x5 * x2;
  return x - (x3 / 6.0f) + (x5 / 120.0f) - (x7 / 5040.0f);
}

__device__ inline float apply_policy_activation(int code, float x) {
  switch (code) {
    case 0:
      return policy_sigmoid(x);
    case 1:
      return fast_tanh(x);
    case 2:
      return x > 0.0f ? x : 0.0f;
    case 3:
      return x;
    case 4:
      return clampf(x, -1.0f, 1.0f);
    case 5:
      return policy_gauss(x);
    case 6:
      return policy_sin(x);
    case 7:
      return absf(x);
    default:
      return x;
  }
}

extern "C" __global__ void policy_feedforward_kernel(
    const int* activation,
    const float* bias,
    const float* response,
    const int* output_indices,
    const int* source_kind,
    const int* source_index,
    const float* weights,
    const int* incoming_offsets,
    int batch_size,
    int input_count,
    int node_count,
    int output_count,
    const float* inputs,
    float* outputs) {
  if (tid_x() != 0) {
    return;
  }

  int sample_idx = (int)ctaid_x();
  if (sample_idx >= batch_size) {
    return;
  }

  extern __shared__ float shared[];
  float* node_values = shared;
  for (int idx = 0; idx < node_count; ++idx) {
    node_values[idx] = 0.0f;
  }

  int input_base = sample_idx * input_count;
  int output_base = sample_idx * output_count;

  for (int node_idx = 0; node_idx < node_count; ++node_idx) {
    float aggregated = 0.0f;
    int start = incoming_offsets[node_idx];
    int end = incoming_offsets[node_idx + 1];
    for (int edge_idx = start; edge_idx < end; ++edge_idx) {
      float source = 0.0f;
      if (source_kind[edge_idx] == 0) {
        source = inputs[input_base + source_index[edge_idx]];
      } else {
        source = node_values[source_index[edge_idx]];
      }
      aggregated += source * weights[edge_idx];
    }
    float pre = bias[node_idx] + (response[node_idx] * aggregated);
    node_values[node_idx] = apply_policy_activation(activation[node_idx], pre);
  }

  for (int output_idx = 0; output_idx < output_count; ++output_idx) {
    outputs[output_base + output_idx] = node_values[output_indices[output_idx]];
  }
}

extern "C" __global__ void policy_recurrent_kernel(
    const int* activation,
    const float* bias,
    const float* response,
    const unsigned char* memory_gate_enabled,
    const float* memory_gate_bias,
    const float* memory_gate_response,
    const int* output_indices,
    const int* source_kind,
    const int* source_index,
    const float* weights,
    const int* incoming_offsets,
    int batch_size,
    int input_count,
    int node_count,
    int output_count,
    const float* inputs,
    const float* snapshots_in,
    float* outputs,
    float* snapshots_out) {
  if (tid_x() != 0) {
    return;
  }

  int sample_idx = (int)ctaid_x();
  if (sample_idx >= batch_size) {
    return;
  }

  extern __shared__ float shared[];
  float* next_values = shared;
  for (int idx = 0; idx < node_count; ++idx) {
    next_values[idx] = 0.0f;
  }

  int input_base = sample_idx * input_count;
  int snapshot_base = sample_idx * node_count;
  int output_base = sample_idx * output_count;

  for (int node_idx = 0; node_idx < node_count; ++node_idx) {
    float aggregated = 0.0f;
    int start = incoming_offsets[node_idx];
    int end = incoming_offsets[node_idx + 1];
    for (int edge_idx = start; edge_idx < end; ++edge_idx) {
      float source = 0.0f;
      if (source_kind[edge_idx] == 0) {
        source = inputs[input_base + source_index[edge_idx]];
      } else {
        source = snapshots_in[snapshot_base + source_index[edge_idx]];
      }
      aggregated += source * weights[edge_idx];
    }

    float candidate_pre = bias[node_idx] + (response[node_idx] * aggregated);
    float candidate_value = apply_policy_activation(activation[node_idx], candidate_pre);
    if (memory_gate_enabled[node_idx]) {
      float gate_pre = memory_gate_bias[node_idx] + (memory_gate_response[node_idx] * aggregated);
      float gate = policy_sigmoid(gate_pre);
      float previous = snapshots_in[snapshot_base + node_idx];
      next_values[node_idx] = ((1.0f - gate) * previous) + (gate * candidate_value);
    } else {
      next_values[node_idx] = candidate_value;
    }
    snapshots_out[snapshot_base + node_idx] = next_values[node_idx];
  }

  for (int output_idx = 0; output_idx < output_count; ++output_idx) {
    outputs[output_base + output_idx] = next_values[output_indices[output_idx]];
  }
}
