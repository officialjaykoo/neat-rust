use std::error::Error;
use std::fmt;

pub const BUILTIN_ACTIVATIONS: &[&str] = &[
    "sigmoid", "tanh", "sin", "gauss", "relu", "elu", "lelu", "selu", "softplus", "identity",
    "clamped", "inv", "log", "exp", "abs", "hat", "square", "cube",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Sin,
    Gauss,
    Relu,
    Elu,
    Lelu,
    Selu,
    Softplus,
    Clamped,
    Identity,
    Inv,
    Log,
    Exp,
    Abs,
    Hat,
    Square,
    Cube,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationError {
    name: String,
}

impl ActivationFunction {
    pub fn from_name(name: &str) -> Option<Self> {
        match normalize_name(name).as_str() {
            "sigmoid" => Some(Self::Sigmoid),
            "tanh" => Some(Self::Tanh),
            "sin" => Some(Self::Sin),
            "gauss" => Some(Self::Gauss),
            "relu" => Some(Self::Relu),
            "elu" => Some(Self::Elu),
            "lelu" => Some(Self::Lelu),
            "selu" => Some(Self::Selu),
            "softplus" => Some(Self::Softplus),
            "clamped" => Some(Self::Clamped),
            "identity" => Some(Self::Identity),
            "inv" => Some(Self::Inv),
            "log" => Some(Self::Log),
            "exp" => Some(Self::Exp),
            "abs" => Some(Self::Abs),
            "hat" => Some(Self::Hat),
            "square" => Some(Self::Square),
            "cube" => Some(Self::Cube),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Sin => "sin",
            Self::Gauss => "gauss",
            Self::Relu => "relu",
            Self::Elu => "elu",
            Self::Lelu => "lelu",
            Self::Selu => "selu",
            Self::Softplus => "softplus",
            Self::Clamped => "clamped",
            Self::Identity => "identity",
            Self::Inv => "inv",
            Self::Log => "log",
            Self::Exp => "exp",
            Self::Abs => "abs",
            Self::Hat => "hat",
            Self::Square => "square",
            Self::Cube => "cube",
        }
    }

    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid_activation(value),
            Self::Tanh => tanh_activation(value),
            Self::Sin => sin_activation(value),
            Self::Gauss => gauss_activation(value),
            Self::Relu => relu_activation(value),
            Self::Elu => elu_activation(value),
            Self::Lelu => lelu_activation(value),
            Self::Selu => selu_activation(value),
            Self::Softplus => softplus_activation(value),
            Self::Clamped => clamped_activation(value),
            Self::Identity => identity_activation(value),
            Self::Inv => inv_activation(value),
            Self::Log => log_activation(value),
            Self::Exp => exp_activation(value),
            Self::Abs => abs_activation(value),
            Self::Hat => hat_activation(value),
            Self::Square => square_activation(value),
            Self::Cube => cube_activation(value),
        }
    }
}

impl ActivationError {
    pub fn unknown(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ActivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown activation function {:?}", self.name)
    }
}

impl Error for ActivationError {}

pub fn is_valid_activation(name: &str) -> bool {
    ActivationFunction::from_name(name).is_some()
}

pub fn activate(name: &str, value: f64) -> Result<f64, ActivationError> {
    let activation =
        ActivationFunction::from_name(name).ok_or_else(|| ActivationError::unknown(name))?;
    Ok(activation.apply(value))
}

pub fn sigmoid_activation(value: f64) -> f64 {
    let z = clamp(5.0 * value, -60.0, 60.0);
    1.0 / (1.0 + (-z).exp())
}

pub fn tanh_activation(value: f64) -> f64 {
    clamp(2.5 * value, -60.0, 60.0).tanh()
}

pub fn sin_activation(value: f64) -> f64 {
    clamp(5.0 * value, -60.0, 60.0).sin()
}

pub fn gauss_activation(value: f64) -> f64 {
    let z = clamp(value, -3.4, 3.4);
    (-5.0 * z.powi(2)).exp()
}

pub fn relu_activation(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

pub fn elu_activation(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        value.exp() - 1.0
    }
}

pub fn lelu_activation(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        0.005 * value
    }
}

pub fn selu_activation(value: f64) -> f64 {
    const LAMBDA: f64 = 1.0507009873554804934193349852946;
    const ALPHA: f64 = 1.6732632423543772848170429916717;
    if value > 0.0 {
        LAMBDA * value
    } else {
        LAMBDA * ALPHA * (value.exp() - 1.0)
    }
}

pub fn softplus_activation(value: f64) -> f64 {
    let z = clamp(5.0 * value, -60.0, 60.0);
    0.2 * (1.0 + z.exp()).ln()
}

pub fn clamped_activation(value: f64) -> f64 {
    clamp(value, -1.0, 1.0)
}

pub fn identity_activation(value: f64) -> f64 {
    value
}

pub fn inv_activation(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }

    let inverted = 1.0 / value;
    if inverted.is_finite() {
        inverted
    } else {
        0.0
    }
}

pub fn log_activation(value: f64) -> f64 {
    value.max(1e-7).ln()
}

pub fn exp_activation(value: f64) -> f64 {
    clamp(value, -60.0, 60.0).exp()
}

pub fn abs_activation(value: f64) -> f64 {
    value.abs()
}

pub fn hat_activation(value: f64) -> f64 {
    (1.0 - value.abs()).max(0.0)
}

pub fn square_activation(value: f64) -> f64 {
    value.powi(2)
}

pub fn cube_activation(value: f64) -> f64 {
    value.powi(3)
}

fn normalize_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn clamp(value: f64, min_value: f64, max_value: f64) -> f64 {
    value.max(min_value).min(max_value)
}
