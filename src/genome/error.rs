use std::error::Error;
use std::fmt;

use crate::gene::{ConnectionKey, GeneError};
use crate::ids::GenomeId;

#[derive(Debug, Clone, PartialEq)]
pub enum GenomeError {
    Gene(GeneError),
    EmptyChoice(&'static str),
    InvalidConnection(ConnectionKey),
    MissingFitness(GenomeId),
    UnsupportedInitialConnection(String),
}

impl fmt::Display for GenomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gene(err) => write!(f, "{err}"),
            Self::EmptyChoice(name) => write!(f, "cannot choose from empty {name}"),
            Self::InvalidConnection(key) => write!(f, "invalid connection {key}"),
            Self::MissingFitness(key) => write!(f, "missing fitness for genome {key}"),
            Self::UnsupportedInitialConnection(value) => {
                write!(f, "unsupported initial_connection {value}")
            }
        }
    }
}

impl Error for GenomeError {}

impl From<GeneError> for GenomeError {
    fn from(value: GeneError) -> Self {
        Self::Gene(value)
    }
}
