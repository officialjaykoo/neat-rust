use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use crate::config::{
    BoolAttributeConfig, ChoiceAttributeConfig, ChoiceAttributeDefault, ConfigChoice,
    FloatAttributeConfig,
};

pub trait RandomSource {
    fn next_f64(&mut self) -> f64;

    fn next_bool(&mut self, probability: f64) -> bool {
        if probability <= 0.0 {
            return false;
        }
        if probability >= 1.0 {
            return true;
        }
        self.next_f64() < probability
    }

    fn next_index(&mut self, upper_exclusive: usize) -> Option<usize> {
        if upper_exclusive == 0 {
            return None;
        }
        let raw = self.next_f64();
        let index = (raw * upper_exclusive as f64).floor() as usize;
        Some(index.min(upper_exclusive - 1))
    }

    fn next_gaussian(&mut self, mean: f64, stdev: f64) -> f64 {
        if stdev == 0.0 {
            return mean;
        }

        let u1 = self.next_f64().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let u2 = self.next_f64().clamp(0.0, 1.0 - f64::EPSILON);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + (stdev * z0)
    }

    fn next_uniform(&mut self, min_value: f64, max_value: f64) -> f64 {
        min_value + ((max_value - min_value) * self.next_f64())
    }
}

#[derive(Debug, Clone)]
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    pub fn from_state(state: u64) -> Self {
        Self::seed_from_u64(state)
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(2_685_821_657_736_338_717)
    }
}

impl RandomSource for XorShiftRng {
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeError {
    InvalidFloatBounds {
        min_value: String,
        max_value: String,
    },
    EmptyChoiceOptions,
    InvalidChoiceDefault {
        default: String,
        options: Vec<String>,
    },
}

impl fmt::Display for AttributeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFloatBounds {
                min_value,
                max_value,
            } => write!(
                f,
                "invalid float attribute bounds: min_value={min_value}, max_value={max_value}"
            ),
            Self::EmptyChoiceOptions => write!(f, "choice attribute options cannot be empty"),
            Self::InvalidChoiceDefault { default, options } => write!(
                f,
                "invalid choice attribute default {default:?}; expected one of: {}",
                options.join(" ")
            ),
        }
    }
}

impl Error for AttributeError {}

pub struct FloatAttribute;

impl FloatAttribute {
    pub fn validate(config: &FloatAttributeConfig) -> Result<(), AttributeError> {
        if config.max_value < config.min_value {
            return Err(AttributeError::InvalidFloatBounds {
                min_value: config.min_value.to_string(),
                max_value: config.max_value.to_string(),
            });
        }
        Ok(())
    }

    pub fn clamp(value: f64, config: &FloatAttributeConfig) -> f64 {
        value.clamp(config.min_value, config.max_value)
    }

    pub fn init_value(
        config: &FloatAttributeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<f64, AttributeError> {
        Self::validate(config)?;
        match &config.init_type {
            crate::config::FloatInitType::Gaussian => Ok(Self::clamp(
                rng.next_gaussian(config.init_mean, config.init_stdev),
                config,
            )),
            crate::config::FloatInitType::Uniform => {
                let min_value = config
                    .min_value
                    .max(config.init_mean - (2.0 * config.init_stdev));
                let max_value = config
                    .max_value
                    .min(config.init_mean + (2.0 * config.init_stdev));
                Ok(rng.next_uniform(min_value, max_value))
            }
        }
    }

    pub fn mutate_value(
        value: f64,
        config: &FloatAttributeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<f64, AttributeError> {
        Self::validate(config)?;
        let roll = rng.next_f64();
        if roll < config.mutate_rate.value() {
            let delta = rng.next_gaussian(0.0, config.mutate_power);
            return Ok(Self::clamp(value + delta, config));
        }

        if roll < config.mutate_rate.value() + config.replace_rate.value() {
            return Self::init_value(config, rng);
        }

        Ok(value)
    }
}

pub struct BoolAttribute;

impl BoolAttribute {
    pub fn init_value(config: &BoolAttributeConfig, _rng: &mut impl RandomSource) -> bool {
        config.default
    }

    pub fn mutate_value(
        value: bool,
        config: &BoolAttributeConfig,
        rng: &mut impl RandomSource,
    ) -> bool {
        let mut mutate_rate = config.mutate_rate.value();
        if value {
            mutate_rate += config.rate_to_false_add.value();
        } else {
            mutate_rate += config.rate_to_true_add.value();
        }

        if rng.next_bool(mutate_rate) {
            return rng.next_bool(0.5);
        }

        value
    }
}

pub struct ChoiceAttribute;

impl ChoiceAttribute {
    pub fn validate<T: ConfigChoice>(
        config: &ChoiceAttributeConfig<T>,
    ) -> Result<(), AttributeError> {
        if config.options.is_empty() {
            return Err(AttributeError::EmptyChoiceOptions);
        }

        if let ChoiceAttributeDefault::Value(default) = config.default {
            if !config.options.contains(&default) {
                return Err(AttributeError::InvalidChoiceDefault {
                    default: default.name().to_string(),
                    options: config
                        .options
                        .iter()
                        .map(|option| option.name().to_string())
                        .collect(),
                });
            }
        }
        Ok(())
    }

    pub fn init_value<T: ConfigChoice>(
        config: &ChoiceAttributeConfig<T>,
        rng: &mut impl RandomSource,
    ) -> Result<T, AttributeError> {
        Self::validate(config)?;
        match config.default {
            ChoiceAttributeDefault::Random => choose_option(&config.options, rng),
            ChoiceAttributeDefault::Value(value) => Ok(value),
        }
    }

    pub fn mutate_value<T: ConfigChoice>(
        value: T,
        config: &ChoiceAttributeConfig<T>,
        rng: &mut impl RandomSource,
    ) -> Result<T, AttributeError> {
        Self::validate(config)?;
        if rng.next_bool(config.mutate_rate.value()) {
            return choose_option(&config.options, rng);
        }

        Ok(value)
    }
}

fn choose_option<T: Copy>(options: &[T], rng: &mut impl RandomSource) -> Result<T, AttributeError> {
    let index = rng
        .next_index(options.len())
        .ok_or(AttributeError::EmptyChoiceOptions)?;
    Ok(options[index])
}
