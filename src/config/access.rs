use std::collections::{BTreeMap, BTreeSet};

use super::{
    BoolAttributeConfig, ConfigError, FloatAttributeConfig, FloatInitType, IniDocument,
    StringAttributeConfig,
};

pub(super) fn require_section<'a>(
    ini: &'a IniDocument,
    section: &str,
) -> Result<&'a BTreeMap<String, String>, ConfigError> {
    ini.get(section)
        .ok_or_else(|| ConfigError::MissingSection(section.to_string()))
}

fn get_raw<'a>(
    section_map: &'a BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<&'a str, ConfigError> {
    section_map
        .get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| ConfigError::MissingKey {
            section: section.to_string(),
            key: key.to_string(),
        })
}

pub(super) fn get_string(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<String, ConfigError> {
    Ok(get_raw(section_map, section, key)?.to_string())
}

pub(super) fn get_optional_string_default(
    section_map: &BTreeMap<String, String>,
    _section: &str,
    key: &str,
    default: &str,
) -> Result<String, ConfigError> {
    Ok(section_map
        .get(key)
        .map(|value| value.to_string())
        .unwrap_or_else(|| default.to_string()))
}

pub(super) fn get_usize(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<usize, ConfigError> {
    let raw = get_raw(section_map, section, key)?;
    raw.parse::<usize>()
        .map_err(|err| ConfigError::InvalidValue {
            section: section.to_string(),
            key: key.to_string(),
            value: raw.to_string(),
            message: err.to_string(),
        })
}

pub(super) fn get_optional_u64(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<Option<u64>, ConfigError> {
    let Some(raw) = section_map.get(key) else {
        return Ok(None);
    };
    raw.parse::<u64>()
        .map(Some)
        .map_err(|err| ConfigError::InvalidValue {
            section: section.to_string(),
            key: key.to_string(),
            value: raw.to_string(),
            message: err.to_string(),
        })
}

pub(super) fn get_f64(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<f64, ConfigError> {
    let raw = get_raw(section_map, section, key)?;
    raw.parse::<f64>().map_err(|err| ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: raw.to_string(),
        message: err.to_string(),
    })
}

pub(super) fn get_optional_f64_default(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
    default: f64,
) -> Result<f64, ConfigError> {
    let Some(raw) = section_map.get(key) else {
        return Ok(default);
    };
    raw.parse::<f64>().map_err(|err| ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: raw.to_string(),
        message: err.to_string(),
    })
}

pub(super) fn get_bool(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
) -> Result<bool, ConfigError> {
    let raw = get_raw(section_map, section, key)?;
    parse_bool(raw).ok_or_else(|| ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: raw.to_string(),
        message: "expected true/false, yes/no, on/off, or 1/0".to_string(),
    })
}

pub(super) fn get_optional_bool_default(
    section_map: &BTreeMap<String, String>,
    section: &str,
    key: &str,
    default: bool,
) -> Result<bool, ConfigError> {
    let Some(raw) = section_map.get(key) else {
        return Ok(default);
    };
    parse_bool(raw).ok_or_else(|| ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: raw.to_string(),
        message: "expected true/false, yes/no, on/off, or 1/0".to_string(),
    })
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    }
}

pub(super) fn get_string_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
) -> Result<StringAttributeConfig, ConfigError> {
    let default = get_string(section_map, section, &format!("{prefix}_default"))?;
    let mutate_rate = get_f64(section_map, section, &format!("{prefix}_mutate_rate"))?;
    let options = split_words(&get_string(
        section_map,
        section,
        &format!("{prefix}_options"),
    )?);
    validate_default_in_options(section, &format!("{prefix}_default"), &default, &options)?;
    Ok(StringAttributeConfig {
        default,
        mutate_rate,
        options,
    })
}

pub(super) fn get_optional_string_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
    default: StringAttributeConfig,
) -> Result<StringAttributeConfig, ConfigError> {
    if !section_map.contains_key(&format!("{prefix}_default")) {
        return Ok(default);
    }
    get_string_attribute(section_map, section, prefix)
}

pub(super) fn get_float_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
) -> Result<FloatAttributeConfig, ConfigError> {
    Ok(FloatAttributeConfig {
        init_mean: get_f64(section_map, section, &format!("{prefix}_init_mean"))?,
        init_stdev: get_f64(section_map, section, &format!("{prefix}_init_stdev"))?,
        init_type: FloatInitType::from_raw(&get_string(
            section_map,
            section,
            &format!("{prefix}_init_type"),
        )?),
        max_value: get_f64(section_map, section, &format!("{prefix}_max_value"))?,
        min_value: get_f64(section_map, section, &format!("{prefix}_min_value"))?,
        mutate_power: get_f64(section_map, section, &format!("{prefix}_mutate_power"))?,
        mutate_rate: get_f64(section_map, section, &format!("{prefix}_mutate_rate"))?,
        replace_rate: get_f64(section_map, section, &format!("{prefix}_replace_rate"))?,
    })
}

pub(super) fn get_optional_float_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
    default: FloatAttributeConfig,
) -> Result<FloatAttributeConfig, ConfigError> {
    if !section_map.contains_key(&format!("{prefix}_init_mean")) {
        return Ok(default);
    }
    get_float_attribute(section_map, section, prefix)
}

pub(super) fn get_bool_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
) -> Result<BoolAttributeConfig, ConfigError> {
    Ok(BoolAttributeConfig {
        default: get_bool(section_map, section, &format!("{prefix}_default"))?,
        mutate_rate: get_f64(section_map, section, &format!("{prefix}_mutate_rate"))?,
        rate_to_true_add: get_f64(section_map, section, &format!("{prefix}_rate_to_true_add"))?,
        rate_to_false_add: get_f64(section_map, section, &format!("{prefix}_rate_to_false_add"))?,
    })
}

pub(super) fn get_optional_bool_attribute(
    section_map: &BTreeMap<String, String>,
    section: &str,
    prefix: &str,
    default: BoolAttributeConfig,
) -> Result<BoolAttributeConfig, ConfigError> {
    if !section_map.contains_key(&format!("{prefix}_default")) {
        return Ok(default);
    }
    get_bool_attribute(section_map, section, prefix)
}

fn split_words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn validate_default_in_options(
    section: &str,
    key: &str,
    default: &str,
    options: &[String],
) -> Result<(), ConfigError> {
    if matches!(
        default.trim().to_ascii_lowercase().as_str(),
        "none" | "random"
    ) {
        return Ok(());
    }

    let option_set: BTreeSet<&str> = options.iter().map(String::as_str).collect();
    if option_set.contains(default) {
        Ok(())
    } else {
        Err(ConfigError::InvalidValue {
            section: section.to_string(),
            key: key.to_string(),
            value: default.to_string(),
            message: format!("default must be one of: {}", options.join(" ")),
        })
    }
}
