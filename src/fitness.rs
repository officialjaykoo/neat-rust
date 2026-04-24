use std::error::Error;
use std::fmt;

use crate::config::FitnessCriterion;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FitnessScore(f64);

impl FitnessScore {
    pub fn new(value: f64) -> Result<Self, FitnessScoreError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(FitnessScoreError::NonFinite(value))
        }
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn is_better_than(self, other: Self, criterion: &FitnessCriterion) -> bool {
        criterion.is_better(self.0, other.0)
    }
}

impl TryFrom<f64> for FitnessScore {
    type Error = FitnessScoreError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<FitnessScore> for f64 {
    fn from(value: FitnessScore) -> Self {
        value.0
    }
}

impl fmt::Display for FitnessScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitnessScoreError {
    NonFinite(f64),
}

impl fmt::Display for FitnessScoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(value) => write!(f, "fitness must be finite, got {value}"),
        }
    }
}

impl Error for FitnessScoreError {}
