use neat_rust::{
    activate, clamped_activation, identity_activation, is_valid_activation, relu_activation,
    tanh_activation, ActivationFunction,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn resolves_required_activation_functions() {
    assert_eq!(
        ActivationFunction::from_name("tanh"),
        Some(ActivationFunction::Tanh)
    );
    assert_eq!(
        ActivationFunction::from_name(" ReLU "),
        Some(ActivationFunction::Relu)
    );
    assert_eq!(
        ActivationFunction::from_name("clamped"),
        Some(ActivationFunction::Clamped)
    );
    assert_eq!(
        ActivationFunction::from_name("identity"),
        Some(ActivationFunction::Identity)
    );
    assert!(!is_valid_activation("missing"));
}

#[test]
fn applies_required_activation_functions() {
    assert_close(tanh_activation(1.0), 2.5f64.tanh());
    assert_close(activate("tanh", 100.0).expect("tanh should resolve"), 1.0);
    assert_close(relu_activation(-2.0), 0.0);
    assert_close(relu_activation(3.5), 3.5);
    assert_close(clamped_activation(-2.0), -1.0);
    assert_close(clamped_activation(0.25), 0.25);
    assert_close(clamped_activation(2.0), 1.0);
    assert_close(identity_activation(-7.0), -7.0);
}
