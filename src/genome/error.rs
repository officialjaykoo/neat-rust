use std::error::Error;
use std::fmt;

use crate::gene::{ConnectionKey, GeneError, NodeKey};
use crate::ids::GenomeId;

#[derive(Debug, Clone, PartialEq)]
pub enum GenomeError {
    Gene(GeneError),
    EmptyChoice(&'static str),
    InvalidConnection(ConnectionKey),
    ConnectionIntoInputNode {
        genome_key: GenomeId,
        connection_key: ConnectionKey,
    },
    ConnectionFromUnknownNode {
        genome_key: GenomeId,
        connection_key: ConnectionKey,
        node_key: NodeKey,
    },
    ConnectionToUnknownNode {
        genome_key: GenomeId,
        connection_key: ConnectionKey,
        node_key: NodeKey,
    },
    FeedForwardCycle {
        genome_key: GenomeId,
        connection_key: ConnectionKey,
    },
    InputNodeStored {
        genome_key: GenomeId,
        node_key: NodeKey,
    },
    MissingOutputNode {
        genome_key: GenomeId,
        node_key: NodeKey,
    },
    InvalidFitness {
        genome_key: GenomeId,
        value: f64,
    },
    MissingFitness(GenomeId),
    UnsupportedInitialConnection(String),
}

impl fmt::Display for GenomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gene(err) => write!(f, "{err}"),
            Self::EmptyChoice(name) => write!(f, "cannot choose from empty {name}"),
            Self::InvalidConnection(key) => write!(f, "invalid connection {key}"),
            Self::ConnectionIntoInputNode {
                genome_key,
                connection_key,
            } => write!(
                f,
                "genome {genome_key} has connection into input node: {connection_key}"
            ),
            Self::ConnectionFromUnknownNode {
                genome_key,
                connection_key,
                node_key,
            } => write!(
                f,
                "genome {genome_key} connection {connection_key} starts at unknown node {node_key}"
            ),
            Self::ConnectionToUnknownNode {
                genome_key,
                connection_key,
                node_key,
            } => write!(
                f,
                "genome {genome_key} connection {connection_key} targets unknown node {node_key}"
            ),
            Self::FeedForwardCycle {
                genome_key,
                connection_key,
            } => write!(
                f,
                "genome {genome_key} feed-forward topology would cycle at {connection_key}"
            ),
            Self::InputNodeStored {
                genome_key,
                node_key,
            } => write!(
                f,
                "genome {genome_key} stores input node {node_key}; input nodes are implicit"
            ),
            Self::MissingOutputNode {
                genome_key,
                node_key,
            } => write!(f, "genome {genome_key} is missing output node {node_key}"),
            Self::InvalidFitness { genome_key, value } => {
                write!(f, "genome {genome_key} has non-finite fitness {value}")
            }
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
