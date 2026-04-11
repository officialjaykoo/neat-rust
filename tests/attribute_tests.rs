use neat_rust::{
    BoolAttribute, BoolAttributeConfig, FloatAttribute, FloatAttributeConfig, RandomSource,
    StringAttribute, StringAttributeConfig,
};

#[derive(Debug, Clone)]
struct SequenceRng {
    values: Vec<f64>,
    index: usize,
}

impl SequenceRng {
    fn new(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            index: 0,
        }
    }
}

impl RandomSource for SequenceRng {
    fn next_f64(&mut self) -> f64 {
        let value = *self
            .values
            .get(self.index)
            .expect("test RNG sequence should have enough values");
        self.index += 1;
        value
    }
}

fn float_config() -> FloatAttributeConfig {
    FloatAttributeConfig {
        init_mean: 1.0,
        init_stdev: 1.0,
        init_type: "uniform".to_string(),
        max_value: 10.0,
        min_value: 0.0,
        mutate_power: 0.25,
        mutate_rate: 1.0,
        replace_rate: 0.0,
    }
}

fn string_config(default: &str, mutate_rate: f64) -> StringAttributeConfig {
    StringAttributeConfig {
        default: default.to_string(),
        mutate_rate,
        options: vec![
            "tanh".to_string(),
            "relu".to_string(),
            "clamped".to_string(),
            "identity".to_string(),
        ],
    }
}

#[test]
fn initializes_float_with_uniform_bounds() {
    let mut rng = SequenceRng::new(&[0.25]);
    let value = FloatAttribute::init_value(&float_config(), &mut rng)
        .expect("uniform float should initialize");

    assert_eq!(value, 0.75);
}

#[test]
fn mutates_float_with_gaussian_delta_and_clamp() {
    let mut config = float_config();
    config.init_type = "gaussian".to_string();
    let mut rng = SequenceRng::new(&[0.0, (-0.5f64).exp(), 0.0]);
    let value = FloatAttribute::mutate_value(1.0, &config, &mut rng).expect("float should mutate");

    assert!((value - 1.25).abs() < 1e-12, "value={value}");
}

#[test]
fn mutates_bool_with_directional_rate() {
    let config = BoolAttributeConfig {
        default: false,
        mutate_rate: 0.0,
        rate_to_true_add: 1.0,
        rate_to_false_add: 0.0,
    };
    let mut rng = SequenceRng::new(&[0.25]);

    assert!(BoolAttribute::mutate_value(false, &config, &mut rng));
}

#[test]
fn initializes_and_mutates_string_from_options() {
    let mut init_rng = SequenceRng::new(&[0.75]);
    let initial = StringAttribute::init_value(&string_config("random", 0.0), &mut init_rng)
        .expect("random string init should choose an option");
    assert_eq!(initial, "identity");

    let mut mutate_rng = SequenceRng::new(&[0.0, 0.5]);
    let mutated =
        StringAttribute::mutate_value("tanh", &string_config("tanh", 0.75), &mut mutate_rng)
            .expect("string should mutate");
    assert_eq!(mutated, "clamped");
}
