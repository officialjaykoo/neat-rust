use crate::activation::{ActivationError, ActivationFunction};
use crate::aggregation::{AggregationError, AggregationFunction};
use crate::attributes::RandomSource;

use super::GeneError;

pub(super) fn choose_copy<T: Copy>(first: T, second: T, rng: &mut impl RandomSource) -> T {
    if rng.next_f64() > 0.5 {
        first
    } else {
        second
    }
}

pub(super) fn parse_activation(name: String) -> Result<ActivationFunction, GeneError> {
    ActivationFunction::from_name(&name).ok_or_else(|| ActivationError::unknown(&name).into())
}

pub(super) fn parse_aggregation(name: String) -> Result<AggregationFunction, GeneError> {
    AggregationFunction::from_name(&name).ok_or_else(|| AggregationError::unknown(&name).into())
}
