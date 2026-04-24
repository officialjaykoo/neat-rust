use std::error::Error;
use std::fmt;

use crate::activation::ActivationError;
use crate::aggregation::AggregationError;
use crate::attributes::AttributeError;

#[derive(Debug, Clone, PartialEq)]
pub enum GeneError {
    Attribute(AttributeError),
    Activation(ActivationError),
    Aggregation(AggregationError),
    InnovationMismatch { left: i64, right: i64 },
    KeyMismatch { left: String, right: String },
}

impl fmt::Display for GeneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attribute(err) => write!(f, "{err}"),
            Self::Activation(err) => write!(f, "{err}"),
            Self::Aggregation(err) => write!(f, "{err}"),
            Self::InnovationMismatch { left, right } => {
                write!(f, "gene innovation mismatch: left={left}, right={right}")
            }
            Self::KeyMismatch { left, right } => {
                write!(f, "gene key mismatch: left={left}, right={right}")
            }
        }
    }
}

impl Error for GeneError {}

impl From<AttributeError> for GeneError {
    fn from(value: AttributeError) -> Self {
        Self::Attribute(value)
    }
}

impl From<ActivationError> for GeneError {
    fn from(value: ActivationError) -> Self {
        Self::Activation(value)
    }
}

impl From<AggregationError> for GeneError {
    fn from(value: AggregationError) -> Self {
        Self::Aggregation(value)
    }
}
