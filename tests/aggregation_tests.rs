use neat_rust::{aggregate, is_valid_aggregation, sum_aggregation, AggregationFunction};

#[test]
fn resolves_sum_aggregation() {
    assert_eq!(
        AggregationFunction::from_name(" sum "),
        Some(AggregationFunction::Sum)
    );
    assert!(is_valid_aggregation("mean"));
}

#[test]
fn sums_values() {
    assert_eq!(sum_aggregation(&[]), 0.0);
    assert_eq!(sum_aggregation(&[1.0, -2.5, 4.0]), 2.5);
    assert_eq!(
        aggregate("sum", &[0.25, 0.25, 0.5]).expect("sum should resolve"),
        1.0
    );
}
