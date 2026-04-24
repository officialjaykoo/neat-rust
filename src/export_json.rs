use crate::config::GenomeConfig;
use crate::genome::{input_keys, output_keys, DefaultGenome};

pub const NEAT_PYTHON_GENOME_FORMAT: &str = "neat_python_genome_v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenomeJsonOptions {
    pub feature_profile: String,
}

impl GenomeJsonOptions {
    pub fn new(feature_profile: impl Into<String>) -> Self {
        Self {
            feature_profile: feature_profile.into(),
        }
    }
}

pub fn export_genome_json(
    genome: &DefaultGenome,
    config: &GenomeConfig,
    options: &GenomeJsonOptions,
) -> String {
    let network_type = if config.feed_forward {
        "feedforward"
    } else {
        "recurrent"
    };
    let feature_profile = if options.feature_profile.trim().is_empty() {
        infer_feature_profile(config)
    } else {
        options.feature_profile.trim().to_ascii_lowercase()
    };
    let runtime_memory_profile = if matches!(
        feature_profile.as_str(),
        "material10_v4" | "hand11_v4" | "memory8"
    ) {
        "recent_play_v4"
    } else {
        "none"
    };
    let runtime_memory_slots = if runtime_memory_profile == "recent_play_v4" {
        3
    } else {
        0
    };

    let mut out = String::new();
    out.push('{');
    push_json_string_field(&mut out, "format_version", NEAT_PYTHON_GENOME_FORMAT);
    out.push(',');
    push_json_string_field(&mut out, "network_type", network_type);
    out.push(',');
    push_json_string_field(&mut out, "runtime_memory_profile", runtime_memory_profile);
    out.push(',');
    push_json_usize_field(&mut out, "runtime_memory_slots", runtime_memory_slots);
    out.push(',');
    push_json_i64_array_field(&mut out, "input_keys", &input_keys(config));
    out.push(',');
    push_json_i64_array_field(&mut out, "output_keys", &output_keys(config));
    out.push(',');
    out.push_str("\"feature_spec\":{");
    push_json_string_field(&mut out, "profile", &feature_profile);
    out.push(',');
    push_json_usize_field(&mut out, "base_features", config.num_inputs);
    out.push('}');
    out.push(',');
    push_nodes(&mut out, genome);
    out.push(',');
    push_connections(&mut out, genome);
    out.push('}');
    out
}

fn infer_feature_profile(config: &GenomeConfig) -> String {
    match config.num_inputs {
        7 => "hand7".to_string(),
        8 => "memory8".to_string(),
        10 => "material10".to_string(),
        11 => "hand11_v4".to_string(),
        _ => format!("input{}", config.num_inputs),
    }
}

fn push_nodes(out: &mut String, genome: &DefaultGenome) {
    out.push_str("\"nodes\":{");
    for (idx, (node_id, node)) in genome.nodes.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        push_json_string(out, &node_id.to_string());
        out.push_str(":{");
        push_json_i64_field(out, "node_id", *node_id);
        out.push(',');
        push_json_string_field(out, "activation", node.activation.name());
        out.push(',');
        push_json_string_field(out, "aggregation", node.aggregation.name());
        out.push(',');
        push_json_f64_field(out, "bias", node.bias);
        out.push(',');
        push_json_f64_field(out, "response", node.response);
        out.push(',');
        push_json_f64_field(out, "time_constant", node.time_constant);
        out.push(',');
        push_json_f64_field(out, "a", node.iz_a);
        out.push(',');
        push_json_f64_field(out, "b", node.iz_b);
        out.push(',');
        push_json_f64_field(out, "c", node.iz_c);
        out.push(',');
        push_json_f64_field(out, "d", node.iz_d);
        out.push(',');
        push_json_bool_field(out, "memory_gate_enabled", node.memory_gate_enabled);
        out.push(',');
        push_json_f64_field(out, "memory_gate_bias", node.memory_gate_bias);
        out.push(',');
        push_json_f64_field(out, "memory_gate_response", node.memory_gate_response);
        out.push('}');
    }
    out.push('}');
}

fn push_connections(out: &mut String, genome: &DefaultGenome) {
    out.push_str("\"connections\":[");
    for (idx, connection) in genome.connections.values().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        push_json_i64_field(out, "in_node", connection.key.input);
        out.push(',');
        push_json_i64_field(out, "out_node", connection.key.output);
        out.push(',');
        push_json_f64_field(out, "weight", connection.weight);
        out.push(',');
        push_json_bool_field(out, "enabled", connection.enabled);
        out.push('}');
    }
    out.push(']');
}

fn push_json_string_field(out: &mut String, key: &str, value: &str) {
    push_json_string(out, key);
    out.push(':');
    push_json_string(out, value);
}

fn push_json_i64_field(out: &mut String, key: &str, value: i64) {
    push_json_string(out, key);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_json_usize_field(out: &mut String, key: &str, value: usize) {
    push_json_string(out, key);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_json_f64_field(out: &mut String, key: &str, value: f64) {
    push_json_string(out, key);
    out.push(':');
    if value.is_finite() {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
}

fn push_json_bool_field(out: &mut String, key: &str, value: bool) {
    push_json_string(out, key);
    out.push(':');
    out.push_str(if value { "true" } else { "false" });
}

fn push_json_i64_array_field(out: &mut String, key: &str, values: &[i64]) {
    push_json_string(out, key);
    out.push(':');
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}
