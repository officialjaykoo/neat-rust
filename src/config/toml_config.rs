use serde::Deserialize;

use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;

use super::{
    AdaptiveMutationConfig, BoolAttributeConfig, ChoiceAttributeConfig, ChoiceAttributeDefault,
    CompatibilityExcessCoefficient, Config, ConfigChoice, ConfigError, FitnessCriterion,
    FitnessSharingMode, FloatAttributeConfig, FloatInitType, GenomeConfig, InitialConnection,
    InitialConnectionMode, MutationRateCaps, NeatConfig, NodeGruTopology, NodeHebbianRule,
    NodeMemoryKind, Probability, ReproductionConfig, SpawnMethod, SpeciesFitnessFunction,
    SpeciesSetConfig, StagnationConfig, StructuralMutationSurer, TargetNumSpecies,
};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TomlConfigDocument {
    neat: RawNeatConfig,
    genome: RawGenomeConfig,
    species_set: RawSpeciesSetConfig,
    stagnation: RawStagnationConfig,
    reproduction: RawReproductionConfig,
}

impl TomlConfigDocument {
    pub(super) fn from_str(text: &str) -> Result<Self, ConfigError> {
        toml::from_str(text).map_err(|err| ConfigError::Parse {
            line: err
                .span()
                .map(|span| byte_offset_to_line(text, span.start))
                .unwrap_or(1),
            message: err.to_string(),
        })
    }

    pub(super) fn into_config(self) -> Result<Config, ConfigError> {
        Ok(Config {
            neat: self.neat.into_config()?,
            genome: self.genome.into_config()?,
            species_set: self.species_set.into_config()?,
            stagnation: self.stagnation.into_config()?,
            reproduction: self.reproduction.into_config()?,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNeatConfig {
    fitness_criterion: String,
    fitness_threshold: f64,
    pop_size: usize,
    reset_on_extinction: bool,
    no_fitness_termination: bool,
    seed: Option<u64>,
}

impl RawNeatConfig {
    fn into_config(self) -> Result<NeatConfig, ConfigError> {
        Ok(NeatConfig {
            fitness_criterion: parse_required(
                "neat",
                "fitness_criterion",
                &self.fitness_criterion,
                FitnessCriterion::parse,
                "max, min, or mean",
            )?,
            fitness_threshold: self.fitness_threshold,
            pop_size: self.pop_size,
            reset_on_extinction: self.reset_on_extinction,
            no_fitness_termination: self.no_fitness_termination,
            seed: self.seed,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGenomeConfig {
    num_inputs: usize,
    num_outputs: usize,
    num_hidden: usize,
    feed_forward: bool,
    initial_connection: RawInitialConnection,
    conn_add_prob: Probability,
    conn_delete_prob: Probability,
    node_add_prob: Probability,
    node_delete_prob: Probability,
    single_structural_mutation: bool,
    structural_mutation_surer: String,
    activation: Option<RawChoiceAttribute>,
    aggregation: Option<RawChoiceAttribute>,
    bias: RawFloatAttribute,
    response: Option<RawFloatAttribute>,
    time_constant: Option<RawFloatAttribute>,
    a: Option<RawFloatAttribute>,
    b: Option<RawFloatAttribute>,
    c: Option<RawFloatAttribute>,
    d: Option<RawFloatAttribute>,
    node_memory_kind: Option<RawChoiceAttribute>,
    node_gru_topology: Option<RawChoiceAttribute>,
    node_gru_reset_bias: Option<RawFloatAttribute>,
    node_gru_reset_response: Option<RawFloatAttribute>,
    node_gru_reset_memory_weight: Option<RawFloatAttribute>,
    node_gru_update_bias: Option<RawFloatAttribute>,
    node_gru_update_response: Option<RawFloatAttribute>,
    node_gru_update_memory_weight: Option<RawFloatAttribute>,
    node_gru_candidate_memory_weight: Option<RawFloatAttribute>,
    node_hebbian_rule: Option<RawChoiceAttribute>,
    node_hebbian_decay: Option<RawFloatAttribute>,
    node_hebbian_eta: Option<RawFloatAttribute>,
    node_hebbian_key_weight: Option<RawFloatAttribute>,
    node_hebbian_alpha: Option<RawFloatAttribute>,
    node_hebbian_mod_bias: Option<RawFloatAttribute>,
    node_hebbian_mod_response: Option<RawFloatAttribute>,
    node_hebbian_theta_decay: Option<RawFloatAttribute>,
    node_linear_decay_bias: Option<RawFloatAttribute>,
    node_linear_decay_response: Option<RawFloatAttribute>,
    node_linear_write_weight: Option<RawFloatAttribute>,
    node_linear_gate_bias: Option<RawFloatAttribute>,
    node_linear_gate_response: Option<RawFloatAttribute>,
    node_linear_min_decay: Option<RawFloatAttribute>,
    node_linear_input_mix: Option<RawFloatAttribute>,
    node_linear_memory_weight: Option<RawFloatAttribute>,
    node_linear_trace_decay: Option<RawFloatAttribute>,
    node_linear_trace_weight: Option<RawFloatAttribute>,
    enabled: RawBoolAttribute,
    compatibility_disjoint_coefficient: f64,
    compatibility_excess_coefficient: Option<String>,
    #[serde(default = "default_true")]
    compatibility_include_node_genes: bool,
    #[serde(default = "default_compatibility_enable_penalty")]
    compatibility_enable_penalty: f64,
    compatibility_weight_coefficient: f64,
    weight: RawFloatAttribute,
}

impl RawGenomeConfig {
    fn into_config(self) -> Result<GenomeConfig, ConfigError> {
        Ok(GenomeConfig {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            num_hidden: self.num_hidden,
            feed_forward: self.feed_forward,
            initial_connection: self.initial_connection.into_config()?,
            conn_add_prob: self.conn_add_prob,
            conn_delete_prob: self.conn_delete_prob,
            node_add_prob: self.node_add_prob,
            node_delete_prob: self.node_delete_prob,
            single_structural_mutation: self.single_structural_mutation,
            structural_mutation_surer: parse_required(
                "genome",
                "structural_mutation_surer",
                &self.structural_mutation_surer,
                StructuralMutationSurer::parse,
                "default, true, or false",
            )?,
            activation: self
                .activation
                .map(|value| {
                    value.into_choice_config(
                        "genome.activation",
                        ActivationFunction::from_name,
                        "built-in activation function",
                    )
                })
                .transpose()?
                .unwrap_or_else(|| ChoiceAttributeConfig {
                    default: ChoiceAttributeDefault::Value(ActivationFunction::Identity),
                    mutate_rate: Probability::zero(),
                    options: vec![ActivationFunction::Identity],
                }),
            aggregation: self
                .aggregation
                .map(|value| {
                    value.into_choice_config(
                        "genome.aggregation",
                        AggregationFunction::from_name,
                        "built-in aggregation function",
                    )
                })
                .transpose()?
                .unwrap_or_else(|| ChoiceAttributeConfig {
                    default: ChoiceAttributeDefault::Value(AggregationFunction::Sum),
                    mutate_rate: Probability::zero(),
                    options: vec![AggregationFunction::Sum],
                }),
            bias: self.bias.into_config()?,
            response: self
                .response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_response_attribute),
            time_constant: self
                .time_constant
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_time_constant_attribute),
            iz_a: self
                .a
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_iz_a_attribute),
            iz_b: self
                .b
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_iz_b_attribute),
            iz_c: self
                .c
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_iz_c_attribute),
            iz_d: self
                .d
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_iz_d_attribute),
            node_memory_kind: self
                .node_memory_kind
                .map(|value| {
                    value.into_choice_config(
                        "genome.node_memory_kind",
                        NodeMemoryKind::parse,
                        "none, node-gru, hebbian, linear-gate, or rg-lru-lite",
                    )
                })
                .transpose()?
                .unwrap_or_else(default_node_memory_kind_attribute),
            node_gru_topology: self
                .node_gru_topology
                .map(|value| {
                    value.into_choice_config(
                        "genome.node_gru_topology",
                        NodeGruTopology::parse,
                        "standard, minimal, coupled, or reset-only",
                    )
                })
                .transpose()?
                .unwrap_or_else(default_node_gru_topology_attribute),
            node_gru_reset_bias: self
                .node_gru_reset_bias
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_bias_attribute),
            node_gru_reset_response: self
                .node_gru_reset_response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_response_attribute),
            node_gru_reset_memory_weight: self
                .node_gru_reset_memory_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_zero_weight_attribute),
            node_gru_update_bias: self
                .node_gru_update_bias
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_bias_attribute),
            node_gru_update_response: self
                .node_gru_update_response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_response_attribute),
            node_gru_update_memory_weight: self
                .node_gru_update_memory_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_zero_weight_attribute),
            node_gru_candidate_memory_weight: self
                .node_gru_candidate_memory_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_zero_weight_attribute),
            node_hebbian_rule: self
                .node_hebbian_rule
                .map(|value| {
                    value.into_choice_config(
                        "genome.node_hebbian_rule",
                        NodeHebbianRule::parse,
                        "plain, oja, bcm, or oja-bcm",
                    )
                })
                .transpose()?
                .unwrap_or_else(default_node_hebbian_rule_attribute),
            node_hebbian_decay: self
                .node_hebbian_decay
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_hebbian_decay_attribute),
            node_hebbian_eta: self
                .node_hebbian_eta
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_hebbian_eta_attribute),
            node_hebbian_key_weight: self
                .node_hebbian_key_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_weight_attribute),
            node_hebbian_alpha: self
                .node_hebbian_alpha
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_hebbian_alpha_attribute),
            node_hebbian_mod_bias: self
                .node_hebbian_mod_bias
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_bias_attribute),
            node_hebbian_mod_response: self
                .node_hebbian_mod_response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_response_attribute),
            node_hebbian_theta_decay: self
                .node_hebbian_theta_decay
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_hebbian_theta_decay_attribute),
            node_linear_decay_bias: self
                .node_linear_decay_bias
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_bias_attribute),
            node_linear_decay_response: self
                .node_linear_decay_response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_response_attribute),
            node_linear_write_weight: self
                .node_linear_write_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_weight_attribute),
            node_linear_gate_bias: self
                .node_linear_gate_bias
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_bias_attribute),
            node_linear_gate_response: self
                .node_linear_gate_response
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_gate_response_attribute),
            node_linear_min_decay: self
                .node_linear_min_decay
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_linear_min_decay_attribute),
            node_linear_input_mix: self
                .node_linear_input_mix
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_zero_weight_attribute),
            node_linear_memory_weight: self
                .node_linear_memory_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_linear_memory_weight_attribute),
            node_linear_trace_decay: self
                .node_linear_trace_decay
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_linear_trace_decay_attribute),
            node_linear_trace_weight: self
                .node_linear_trace_weight
                .map(RawFloatAttribute::into_config)
                .transpose()?
                .unwrap_or_else(default_node_zero_weight_attribute),
            enabled: self.enabled.into_config(),
            compatibility_disjoint_coefficient: self.compatibility_disjoint_coefficient,
            compatibility_excess_coefficient: self
                .compatibility_excess_coefficient
                .map(|value| {
                    parse_required(
                        "genome",
                        "compatibility_excess_coefficient",
                        &value,
                        CompatibilityExcessCoefficient::parse,
                        "auto or a finite number",
                    )
                })
                .transpose()?
                .unwrap_or(CompatibilityExcessCoefficient::Auto),
            compatibility_include_node_genes: self.compatibility_include_node_genes,
            compatibility_enable_penalty: self.compatibility_enable_penalty,
            compatibility_weight_coefficient: self.compatibility_weight_coefficient,
            weight: self.weight.into_config()?,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawInitialConnection {
    Name(String),
    Table {
        mode: String,
        #[serde(default = "default_probability_one")]
        fraction: Probability,
    },
}

impl RawInitialConnection {
    fn into_config(self) -> Result<InitialConnection, ConfigError> {
        match self {
            Self::Name(value) => InitialConnection::parse(&value).ok_or_else(|| {
                invalid_value(
                    "genome",
                    "initial_connection",
                    &value,
                    "unconnected, full, full_direct, full_nodirect, partial, partial_direct, or partial_nodirect",
                )
            }),
            Self::Table { mode, fraction } => {
                let mode_value = match mode.trim().to_ascii_lowercase().as_str() {
                    "unconnected" => InitialConnectionMode::Unconnected,
                    "full_nodirect" => InitialConnectionMode::FullNoDirect,
                    "full_direct" => InitialConnectionMode::FullDirect,
                    "full" => InitialConnectionMode::Full,
                    "partial_nodirect" => InitialConnectionMode::PartialNoDirect,
                    "partial_direct" => InitialConnectionMode::PartialDirect,
                    "partial" => InitialConnectionMode::Partial,
                    _ => {
                        return Err(invalid_value(
                            "genome.initial_connection",
                            "mode",
                            &mode,
                            "known initial connection mode",
                        ));
                    }
                };
                Ok(InitialConnection {
                    mode: mode_value,
                    fraction,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawChoiceAttribute {
    default: String,
    mutate_rate: Probability,
    options: Vec<String>,
}

impl RawChoiceAttribute {
    fn into_choice_config<T: ConfigChoice>(
        self,
        section: &str,
        parse: impl Fn(&str) -> Option<T>,
        expected: &str,
    ) -> Result<ChoiceAttributeConfig<T>, ConfigError> {
        let options = parse_choice_options(section, "options", &self.options, &parse, expected)?;
        let default =
            parse_choice_default(section, "default", &self.default, &options, parse, expected)?;
        Ok(ChoiceAttributeConfig {
            default,
            mutate_rate: self.mutate_rate,
            options,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFloatAttribute {
    init_mean: f64,
    init_stdev: f64,
    init_type: String,
    max_value: f64,
    min_value: f64,
    mutate_power: f64,
    mutate_rate: Probability,
    replace_rate: Probability,
}

impl RawFloatAttribute {
    fn into_config(self) -> Result<FloatAttributeConfig, ConfigError> {
        Ok(FloatAttributeConfig {
            init_mean: self.init_mean,
            init_stdev: self.init_stdev,
            init_type: parse_required(
                "float_attribute",
                "init_type",
                &self.init_type,
                FloatInitType::parse,
                "gaussian or uniform",
            )?,
            max_value: self.max_value,
            min_value: self.min_value,
            mutate_power: self.mutate_power,
            mutate_rate: self.mutate_rate,
            replace_rate: self.replace_rate,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBoolAttribute {
    default: bool,
    mutate_rate: Probability,
    rate_to_true_add: Probability,
    rate_to_false_add: Probability,
}

impl RawBoolAttribute {
    fn into_config(self) -> BoolAttributeConfig {
        BoolAttributeConfig {
            default: self.default,
            mutate_rate: self.mutate_rate,
            rate_to_true_add: self.rate_to_true_add,
            rate_to_false_add: self.rate_to_false_add,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSpeciesSetConfig {
    compatibility_threshold: f64,
    target_num_species: Option<String>,
    #[serde(default = "default_threshold_adjust_rate")]
    threshold_adjust_rate: f64,
    #[serde(default = "default_threshold_min")]
    threshold_min: f64,
    #[serde(default = "default_threshold_max")]
    threshold_max: f64,
}

impl RawSpeciesSetConfig {
    fn into_config(self) -> Result<SpeciesSetConfig, ConfigError> {
        Ok(SpeciesSetConfig {
            compatibility_threshold: self.compatibility_threshold,
            target_num_species: self
                .target_num_species
                .map(|value| {
                    parse_required(
                        "species_set",
                        "target_num_species",
                        &value,
                        TargetNumSpecies::parse,
                        "none or a positive integer",
                    )
                })
                .transpose()?
                .unwrap_or(TargetNumSpecies::Disabled),
            threshold_adjust_rate: self.threshold_adjust_rate,
            threshold_min: self.threshold_min,
            threshold_max: self.threshold_max,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStagnationConfig {
    species_fitness_func: String,
    max_stagnation: usize,
    species_elitism: usize,
}

impl RawStagnationConfig {
    fn into_config(self) -> Result<StagnationConfig, ConfigError> {
        Ok(StagnationConfig {
            species_fitness_func: parse_required(
                "stagnation",
                "species_fitness_func",
                &self.species_fitness_func,
                SpeciesFitnessFunction::parse,
                "mean, max, min, or median",
            )?,
            max_stagnation: self.max_stagnation,
            species_elitism: self.species_elitism,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReproductionConfig {
    elitism: usize,
    survival_threshold: Probability,
    min_species_size: usize,
    fitness_sharing: Option<String>,
    spawn_method: Option<String>,
    #[serde(default)]
    interspecies_crossover_prob: Probability,
    adaptive_mutation: Option<RawAdaptiveMutationConfig>,
}

impl RawReproductionConfig {
    fn into_config(self) -> Result<ReproductionConfig, ConfigError> {
        Ok(ReproductionConfig {
            elitism: self.elitism,
            survival_threshold: self.survival_threshold,
            min_species_size: self.min_species_size,
            fitness_sharing: self
                .fitness_sharing
                .map(|value| {
                    parse_required(
                        "reproduction",
                        "fitness_sharing",
                        &value,
                        FitnessSharingMode::parse,
                        "normalized or canonical",
                    )
                })
                .transpose()?
                .unwrap_or(FitnessSharingMode::Normalized),
            spawn_method: self
                .spawn_method
                .map(|value| {
                    parse_required(
                        "reproduction",
                        "spawn_method",
                        &value,
                        SpawnMethod::parse,
                        "smoothed or proportional",
                    )
                })
                .transpose()?
                .unwrap_or(SpawnMethod::Smoothed),
            interspecies_crossover_prob: self.interspecies_crossover_prob,
            adaptive_mutation: self
                .adaptive_mutation
                .map(RawAdaptiveMutationConfig::into_config)
                .unwrap_or_else(AdaptiveMutationConfig::disabled),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAdaptiveMutationConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    start_after: usize,
    #[serde(default)]
    full_after: usize,
    #[serde(default = "default_adaptive_mutation_max_multiplier")]
    max_multiplier: f64,
    #[serde(default = "default_probability_one")]
    conn_add_prob_cap: Probability,
    #[serde(default = "default_probability_one")]
    conn_delete_prob_cap: Probability,
    #[serde(default = "default_probability_one")]
    node_add_prob_cap: Probability,
    #[serde(default = "default_probability_one")]
    node_delete_prob_cap: Probability,
}

impl RawAdaptiveMutationConfig {
    fn into_config(self) -> AdaptiveMutationConfig {
        AdaptiveMutationConfig {
            enabled: self.enabled,
            start_after: self.start_after,
            full_after: self.full_after,
            max_multiplier: self.max_multiplier.max(1.0),
            caps: MutationRateCaps {
                conn_add_prob: self.conn_add_prob_cap,
                conn_delete_prob: self.conn_delete_prob_cap,
                node_add_prob: self.node_add_prob_cap,
                node_delete_prob: self.node_delete_prob_cap,
            },
        }
    }
}

fn parse_required<T>(
    section: &str,
    key: &str,
    raw: &str,
    parse: impl FnOnce(&str) -> Option<T>,
    expected: &str,
) -> Result<T, ConfigError> {
    parse(raw).ok_or_else(|| invalid_value(section, key, raw, expected))
}

fn parse_choice_options<T: ConfigChoice>(
    section: &str,
    key: &str,
    raw: &[String],
    parse: &impl Fn(&str) -> Option<T>,
    expected: &str,
) -> Result<Vec<T>, ConfigError> {
    let mut out = Vec::new();
    for item in raw {
        let value = parse(item).ok_or_else(|| invalid_value(section, key, item, expected))?;
        if !out.contains(&value) {
            out.push(value);
        }
    }
    if out.is_empty() {
        return Err(invalid_value(section, key, "[]", "non-empty options"));
    }
    Ok(out)
}

fn parse_choice_default<T: ConfigChoice>(
    section: &str,
    key: &str,
    raw: &str,
    options: &[T],
    parse: impl Fn(&str) -> Option<T>,
    expected: &str,
) -> Result<ChoiceAttributeDefault<T>, ConfigError> {
    let trimmed = raw.trim();
    if matches!(trimmed.to_ascii_lowercase().as_str(), "none" | "random") {
        return Ok(ChoiceAttributeDefault::Random);
    }
    let value = parse(trimmed)
        .ok_or_else(|| invalid_value(section, key, raw, &format!("{expected}, random, or none")))?;
    if options.contains(&value) {
        Ok(ChoiceAttributeDefault::Value(value))
    } else {
        Err(invalid_value(
            section,
            key,
            raw,
            &format!(
                "one of: {}",
                options
                    .iter()
                    .map(|value| value.name())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ))
    }
}

fn invalid_value(section: &str, key: &str, value: &str, expected: &str) -> ConfigError {
    ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: value.to_string(),
        message: format!("expected {expected}"),
    }
}

fn byte_offset_to_line(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn default_true() -> bool {
    true
}

fn default_compatibility_enable_penalty() -> f64 {
    1.0
}

fn default_probability_one() -> Probability {
    Probability::one()
}

fn default_adaptive_mutation_max_multiplier() -> f64 {
    1.0
}

fn default_threshold_adjust_rate() -> f64 {
    0.1
}

fn default_threshold_min() -> f64 {
    0.1
}

fn default_threshold_max() -> f64 {
    100.0
}

fn default_float_attribute(value: f64, min_value: f64, max_value: f64) -> FloatAttributeConfig {
    FloatAttributeConfig {
        init_mean: value,
        init_stdev: 0.0,
        init_type: FloatInitType::Gaussian,
        max_value,
        min_value,
        mutate_power: 0.0,
        mutate_rate: Probability::zero(),
        replace_rate: Probability::zero(),
    }
}

fn default_response_attribute() -> FloatAttributeConfig {
    default_float_attribute(1.0, 1.0, 1.0)
}

fn default_time_constant_attribute() -> FloatAttributeConfig {
    default_float_attribute(1.0, 0.01, 10.0)
}

fn default_iz_a_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.02, 0.02, 0.02)
}

fn default_iz_b_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.20, 0.20, 0.20)
}

fn default_iz_c_attribute() -> FloatAttributeConfig {
    default_float_attribute(-65.0, -65.0, -65.0)
}

fn default_iz_d_attribute() -> FloatAttributeConfig {
    default_float_attribute(8.0, 8.0, 8.0)
}

fn default_node_memory_kind_attribute() -> ChoiceAttributeConfig<NodeMemoryKind> {
    ChoiceAttributeConfig {
        default: ChoiceAttributeDefault::Value(NodeMemoryKind::None),
        mutate_rate: Probability::zero(),
        options: vec![NodeMemoryKind::None],
    }
}

fn default_node_gru_topology_attribute() -> ChoiceAttributeConfig<NodeGruTopology> {
    ChoiceAttributeConfig {
        default: ChoiceAttributeDefault::Value(NodeGruTopology::Standard),
        mutate_rate: Probability::zero(),
        options: vec![NodeGruTopology::Standard],
    }
}

fn default_node_hebbian_rule_attribute() -> ChoiceAttributeConfig<NodeHebbianRule> {
    ChoiceAttributeConfig {
        default: ChoiceAttributeDefault::Value(NodeHebbianRule::Oja),
        mutate_rate: Probability::zero(),
        options: vec![NodeHebbianRule::Oja],
    }
}

fn default_node_hebbian_decay_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.9, 0.0, 1.0)
}

fn default_node_hebbian_eta_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.05, -1.0, 1.0)
}

fn default_node_hebbian_alpha_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.5, -5.0, 5.0)
}

fn default_node_hebbian_theta_decay_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.95, 0.0, 0.999)
}

fn default_node_weight_attribute() -> FloatAttributeConfig {
    default_float_attribute(1.0, -5.0, 5.0)
}

fn default_node_zero_weight_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.0, -5.0, 5.0)
}

fn default_node_gate_bias_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.0, -5.0, 5.0)
}

fn default_node_gate_response_attribute() -> FloatAttributeConfig {
    default_float_attribute(1.0, -5.0, 5.0)
}

fn default_node_linear_min_decay_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.5, 0.0, 0.99)
}

fn default_node_linear_memory_weight_attribute() -> FloatAttributeConfig {
    default_float_attribute(1.0, 0.0, 1.5)
}

fn default_node_linear_trace_decay_attribute() -> FloatAttributeConfig {
    default_float_attribute(0.8, 0.0, 0.99)
}
