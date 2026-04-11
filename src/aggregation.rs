use std::error::Error;
use std::fmt;

pub const BUILTIN_AGGREGATIONS: &[&str] =
    &["product", "sum", "max", "min", "maxabs", "median", "mean"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFunction {
    Product,
    Sum,
    Max,
    Min,
    MaxAbs,
    Median,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregationError {
    name: String,
}

impl AggregationFunction {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "product" => Some(Self::Product),
            "sum" => Some(Self::Sum),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "maxabs" => Some(Self::MaxAbs),
            "median" => Some(Self::Median),
            "mean" => Some(Self::Mean),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Product => "product",
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
            Self::MaxAbs => "maxabs",
            Self::Median => "median",
            Self::Mean => "mean",
        }
    }

    pub fn apply(self, values: &[f64]) -> f64 {
        match self {
            Self::Product => product_aggregation(values),
            Self::Sum => sum_aggregation(values),
            Self::Max => max_aggregation(values),
            Self::Min => min_aggregation(values),
            Self::MaxAbs => maxabs_aggregation(values),
            Self::Median => median_aggregation(values),
            Self::Mean => mean_aggregation(values),
        }
    }
}

impl AggregationError {
    pub fn unknown(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for AggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown aggregation function {:?}", self.name)
    }
}

impl Error for AggregationError {}

pub fn is_valid_aggregation(name: &str) -> bool {
    AggregationFunction::from_name(name).is_some()
}

pub fn aggregate(name: &str, values: &[f64]) -> Result<f64, AggregationError> {
    let aggregation =
        AggregationFunction::from_name(name).ok_or_else(|| AggregationError::unknown(name))?;
    Ok(aggregation.apply(values))
}

pub fn sum_aggregation(values: &[f64]) -> f64 {
    values.iter().sum()
}

pub fn product_aggregation(values: &[f64]) -> f64 {
    values.iter().product()
}

pub fn max_aggregation(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

pub fn min_aggregation(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).unwrap_or(0.0)
}

pub fn maxabs_aggregation(values: &[f64]) -> f64 {
    let Some(first) = values.first().copied() else {
        return 0.0;
    };

    let mut best = first;
    for value in values.iter().copied().skip(1) {
        if value.abs() > best.abs() {
            best = value;
        }
    }
    best
}

pub fn median_aggregation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

pub fn mean_aggregation(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        sum_aggregation(values) / values.len() as f64
    }
}
