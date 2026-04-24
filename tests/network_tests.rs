use std::path::PathBuf;

use neat_rust::{
    algorithm::{
        ConnectionKey, DefaultConnectionGene, DefaultGenome, DefaultNodeGene, XorShiftRng,
    },
    io::Config,
    network::{
        Ctrnn, FeedForwardError, FeedForwardNetwork, Iznn, NodeEval, RecurrentError,
        RecurrentNetwork, RecurrentNodeEval,
    },
    prelude::{ActivationFunction, AggregationFunction},
};

fn repo_path(relative: &str) -> PathBuf {
    let relative = relative
        .strip_prefix("scripts/configs/")
        .unwrap_or(relative);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("scripts")
        .join("configs")
        .join(relative)
}

fn key(input: i64, output: i64) -> ConnectionKey {
    ConnectionKey::new(input, output)
}

#[test]
fn feed_forward_network_computes_direct_outputs() {
    let config = Config::from_file(repo_path("scripts/configs/neat_feedforward_memory8.toml"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(3);
    let mut genome = DefaultGenome::new(0);
    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    for node in genome.nodes.values_mut() {
        node.activation = ActivationFunction::Identity;
        node.aggregation = AggregationFunction::Sum;
        node.bias = 0.0;
        node.response = 1.0;
    }
    for connection in genome.connections.values_mut() {
        connection.enabled = false;
        connection.weight = 0.0;
    }
    genome.connections.get_mut(&key(-1, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-1, 0)).unwrap().weight = 2.0;
    genome.connections.get_mut(&key(-2, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-2, 0)).unwrap().weight = -1.0;
    genome.connections.get_mut(&key(-3, 1)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-3, 1)).unwrap().weight = 0.5;

    let mut network =
        FeedForwardNetwork::create(&genome, &config.genome).expect("network should compile");
    let outputs = network
        .activate(&[3.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("network should activate");

    assert_eq!(outputs, vec![2.0, 1.0]);
}

#[test]
fn feed_forward_network_rejects_bad_input_count_like_simple_run() {
    let mut network = FeedForwardNetwork::new(
        vec![-1, -2],
        vec![0],
        vec![NodeEval {
            node: 0,
            activation: ActivationFunction::Identity,
            aggregation: AggregationFunction::Sum,
            bias: 0.0,
            response: 1.0,
            links: vec![(-1, 1.0), (-2, 1.0)],
        }],
    );

    let err = network
        .activate(&[0.5, 0.5, 0.5])
        .expect_err("bad input length must fail");
    assert_eq!(
        err,
        FeedForwardError::InputCountMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn feed_forward_network_handles_orphaned_hidden_node_bias() {
    let config = Config::from_file(repo_path("scripts/configs/neat_feedforward_memory8.toml"))
        .expect("config should parse");
    let mut genome = DefaultGenome::new(1);

    let mut output = DefaultNodeGene::new(0);
    output.bias = 0.5;
    output.response = 1.0;
    output.activation = ActivationFunction::Sigmoid;
    output.aggregation = AggregationFunction::Sum;
    genome.nodes.insert(0, output);

    let mut other_output = DefaultNodeGene::new(1);
    other_output.bias = 0.0;
    other_output.response = 1.0;
    other_output.activation = ActivationFunction::Identity;
    other_output.aggregation = AggregationFunction::Sum;
    genome.nodes.insert(1, other_output);

    let mut hidden = DefaultNodeGene::new(2);
    hidden.bias = 2.0;
    hidden.response = 1.0;
    hidden.activation = ActivationFunction::Sigmoid;
    hidden.aggregation = AggregationFunction::Sum;
    genome.nodes.insert(2, hidden);

    let mut connection = DefaultConnectionGene::with_innovation(key(2, 0), 1);
    connection.weight = 1.0;
    connection.enabled = true;
    genome.connections.insert(connection.key, connection);

    let mut network =
        FeedForwardNetwork::create(&genome, &config.genome).expect("network should compile");
    let outputs = network
        .activate(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("network should activate");

    let orphan_output = 1.0 / (1.0 + (-10.0_f64).exp());
    let expected = 1.0 / (1.0 + (-5.0 * (0.5 + orphan_output)).exp());
    assert!((outputs[0] - expected).abs() < 1.0e-3);
    assert_eq!(outputs[1], 0.0);
}

#[test]
fn recurrent_network_rejects_bad_input_count_like_simple_run() {
    let mut network = RecurrentNetwork::new(
        vec![-1, -2],
        vec![0],
        vec![RecurrentNodeEval {
            node: 0,
            activation: ActivationFunction::Identity,
            aggregation: AggregationFunction::Sum,
            bias: 0.0,
            response: 1.0,
            links: vec![(-1, 1.0), (-2, 1.0)],
            memory_gate_enabled: false,
            memory_gate_bias: 0.0,
            memory_gate_response: 1.0,
        }],
    );

    let err = network
        .activate(&[0.5, 0.5, 0.5])
        .expect_err("bad recurrent input length must fail");
    assert_eq!(
        err,
        RecurrentError::InputCountMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn recurrent_network_uses_previous_state_for_self_loop() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(4);
    let mut genome = DefaultGenome::new(0);
    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    for node in genome.nodes.values_mut() {
        node.activation = ActivationFunction::Identity;
        node.aggregation = AggregationFunction::Sum;
        node.bias = 0.0;
        node.response = 1.0;
        node.memory_gate_enabled = false;
    }
    for connection in genome.connections.values_mut() {
        connection.enabled = false;
        connection.weight = 0.0;
    }
    genome.connections.get_mut(&key(-1, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-1, 0)).unwrap().weight = 1.0;
    genome.connections.get_mut(&key(0, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(0, 0)).unwrap().weight = 0.5;

    let mut network =
        RecurrentNetwork::create(&genome, &config.genome).expect("network should compile");
    let first = network
        .activate(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("first activation should work");
    let second = network
        .activate(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("second activation should work");

    assert_eq!(first[0], 1.0);
    assert_eq!(second[0], 0.5);
}

#[test]
fn ctrnn_uses_exponential_euler_advance() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(5);
    let mut genome = DefaultGenome::new(0);
    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    for node in genome.nodes.values_mut() {
        node.activation = ActivationFunction::Identity;
        node.aggregation = AggregationFunction::Sum;
        node.bias = 0.0;
        node.response = 1.0;
        node.time_constant = 1.0;
    }
    for connection in genome.connections.values_mut() {
        connection.enabled = false;
        connection.weight = 0.0;
    }
    genome.connections.get_mut(&key(-1, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-1, 0)).unwrap().weight = 1.0;

    let mut network = Ctrnn::create(&genome, &config.genome).expect("ctrnn should compile");
    let first = network
        .advance(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, Some(0.1))
        .expect("first advance should work");
    let second = network
        .advance(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, Some(0.1))
        .expect("second advance should work");

    let expected_first = 1.0 - (-0.1_f64).exp();
    let expected_second = 1.0 - (-0.2_f64).exp();
    assert!((first[0] - expected_first).abs() < 1e-12);
    assert!((second[0] - expected_second).abs() < 1e-12);
    assert!((network.time_seconds() - 0.2).abs() < 1e-12);
}

#[test]
fn iznn_spikes_and_resets_like_izhikevich_neuron() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(6);
    let mut genome = DefaultGenome::new(0);
    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    for node in genome.nodes.values_mut() {
        node.bias = 0.0;
        node.iz_a = 0.02;
        node.iz_b = 0.20;
        node.iz_c = -65.0;
        node.iz_d = 8.0;
    }
    for connection in genome.connections.values_mut() {
        connection.enabled = false;
        connection.weight = 0.0;
    }
    genome.connections.get_mut(&key(-1, 0)).unwrap().enabled = true;
    genome.connections.get_mut(&key(-1, 0)).unwrap().weight = 1.0;

    let mut network = Iznn::create(&genome, &config.genome).expect("iznn should compile");
    network
        .set_inputs(&[2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("inputs should set");
    let outputs = network
        .advance(network.get_time_step_msec())
        .expect("advance should work");

    assert_eq!(outputs[0], 1.0);
    let output_neuron = network.neurons.get(&0).expect("output neuron exists");
    assert_eq!(output_neuron.v, -65.0);
    assert!(output_neuron.u > output_neuron.b * output_neuron.v);
}
