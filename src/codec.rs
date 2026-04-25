use std::error::Error;
use std::fmt;

use crate::config::Config;
use crate::genome::DefaultGenome;
use crate::network_impl::{FeedForwardError, FeedForwardNetwork, RecurrentError, RecurrentNetwork};

pub trait GenomeCodec {
    type Phenotype;
    type Error;

    fn decode(
        &self,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<Self::Phenotype, Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkKind {
    FeedForward,
    Recurrent,
}

impl NetworkKind {
    pub fn from_config(config: &Config) -> Self {
        if config.genome.feed_forward {
            Self::FeedForward
        } else {
            Self::Recurrent
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedNetwork {
    FeedForward(FeedForwardNetwork),
    Recurrent(RecurrentNetwork),
}

impl DecodedNetwork {
    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, NetworkCodecError> {
        match self {
            Self::FeedForward(network) => network
                .activate(inputs)
                .map_err(NetworkCodecError::FeedForward),
            Self::Recurrent(network) => network
                .activate(inputs)
                .map_err(NetworkCodecError::Recurrent),
        }
    }

    pub fn reset(&mut self) {
        if let Self::Recurrent(network) = self {
            network.reset();
        }
    }

    pub fn kind(&self) -> NetworkKind {
        match self {
            Self::FeedForward(_) => NetworkKind::FeedForward,
            Self::Recurrent(_) => NetworkKind::Recurrent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkCodec {
    kind: Option<NetworkKind>,
}

impl NetworkCodec {
    pub fn from_config() -> Self {
        Self { kind: None }
    }

    pub fn feed_forward() -> Self {
        Self {
            kind: Some(NetworkKind::FeedForward),
        }
    }

    pub fn recurrent() -> Self {
        Self {
            kind: Some(NetworkKind::Recurrent),
        }
    }

    pub fn with_kind(kind: NetworkKind) -> Self {
        Self { kind: Some(kind) }
    }

    fn resolved_kind(&self, config: &Config) -> NetworkKind {
        self.kind
            .unwrap_or_else(|| NetworkKind::from_config(config))
    }
}

impl Default for NetworkCodec {
    fn default() -> Self {
        Self::from_config()
    }
}

impl GenomeCodec for NetworkCodec {
    type Phenotype = DecodedNetwork;
    type Error = NetworkCodecError;

    fn decode(
        &self,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<Self::Phenotype, Self::Error> {
        match self.resolved_kind(config) {
            NetworkKind::FeedForward => FeedForwardNetwork::create(genome, &config.genome)
                .map(DecodedNetwork::FeedForward)
                .map_err(NetworkCodecError::FeedForward),
            NetworkKind::Recurrent => RecurrentNetwork::create(genome, &config.genome)
                .map(DecodedNetwork::Recurrent)
                .map_err(NetworkCodecError::Recurrent),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkCodecError {
    FeedForward(FeedForwardError),
    Recurrent(RecurrentError),
}

impl fmt::Display for NetworkCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeedForward(err) => write!(f, "{err}"),
            Self::Recurrent(err) => write!(f, "{err}"),
        }
    }
}

impl Error for NetworkCodecError {}
