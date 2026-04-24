use crate::config::GenomeConfig;
use crate::genome::{input_keys, output_keys, DefaultGenome};

mod json_writer;

use json_writer::{push_json_string, JsonObjectWriter};

pub const NEAT_GENOME_FORMAT: &str = "neat_genome_v1";

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
    let runtime_memory_profile = if feature_profile == "memory8" {
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
    let mut root = JsonObjectWriter::new(&mut out);
    root.string_field("format_version", NEAT_GENOME_FORMAT);
    root.string_field("network_type", network_type);
    root.string_field("runtime_memory_profile", runtime_memory_profile);
    root.usize_field("runtime_memory_slots", runtime_memory_slots);
    root.i64_array_field("input_keys", &input_keys(config));
    root.i64_array_field("output_keys", &output_keys(config));
    root.object_field("feature_spec", |feature_spec| {
        feature_spec.string_field("profile", &feature_profile);
        feature_spec.usize_field("base_features", config.num_inputs);
    });
    root.raw_field("nodes", |out| push_nodes_value(out, genome));
    root.raw_field("connections", |out| push_connections_value(out, genome));
    root.finish();
    out
}

fn infer_feature_profile(config: &GenomeConfig) -> String {
    match config.num_inputs {
        7 => "hand7".to_string(),
        8 => "memory8".to_string(),
        10 => "material10".to_string(),
        _ => format!("input{}", config.num_inputs),
    }
}

fn push_nodes_value(out: &mut String, genome: &DefaultGenome) {
    out.push('{');
    for (idx, (node_id, node)) in genome.nodes.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        push_json_string(out, &node_id.to_string());
        out.push(':');
        let mut node_object = JsonObjectWriter::new(out);
        node_object.i64_field("node_id", *node_id);
        node_object.string_field("activation", node.activation.name());
        node_object.string_field("aggregation", node.aggregation.name());
        node_object.f64_field("bias", node.bias);
        node_object.f64_field("response", node.response);
        node_object.f64_field("time_constant", node.time_constant);
        node_object.f64_field("a", node.iz_a);
        node_object.f64_field("b", node.iz_b);
        node_object.f64_field("c", node.iz_c);
        node_object.f64_field("d", node.iz_d);
        node_object.bool_field("memory_gate_enabled", node.memory_gate_enabled);
        node_object.f64_field("memory_gate_bias", node.memory_gate_bias);
        node_object.f64_field("memory_gate_response", node.memory_gate_response);
        node_object.finish();
    }
    out.push('}');
}

fn push_connections_value(out: &mut String, genome: &DefaultGenome) {
    out.push('[');
    for (idx, connection) in genome.connections.values().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let mut connection_object = JsonObjectWriter::new(out);
        connection_object.i64_field("in_node", connection.key.input);
        connection_object.i64_field("out_node", connection.key.output);
        connection_object.f64_field("weight", connection.weight);
        connection_object.bool_field("enabled", connection.enabled);
        connection_object.finish();
    }
    out.push(']');
}
