use std::collections::BTreeSet;

use crate::gene::{ConnectionKey, NodeKey};

pub fn creates_cycle(connections: &[ConnectionKey], test: ConnectionKey) -> bool {
    let input = test.input;
    let output = test.output;
    if input == output {
        return true;
    }

    let mut visited = BTreeSet::new();
    visited.insert(output);

    loop {
        let mut num_added = 0;
        for connection in connections {
            if visited.contains(&connection.input) && !visited.contains(&connection.output) {
                if connection.output == input {
                    return true;
                }
                visited.insert(connection.output);
                num_added += 1;
            }
        }

        if num_added == 0 {
            return false;
        }
    }
}

pub fn required_for_output(
    inputs: &[NodeKey],
    outputs: &[NodeKey],
    connections: &[ConnectionKey],
) -> BTreeSet<NodeKey> {
    let input_set: BTreeSet<NodeKey> = inputs.iter().copied().collect();
    let mut required: BTreeSet<NodeKey> = outputs.iter().copied().collect();
    let mut search: BTreeSet<NodeKey> = outputs.iter().copied().collect();

    loop {
        let mut upstream = BTreeSet::new();
        for connection in connections {
            if search.contains(&connection.output) && !search.contains(&connection.input) {
                upstream.insert(connection.input);
            }
        }

        if upstream.is_empty() {
            break;
        }

        let layer_nodes: BTreeSet<NodeKey> = upstream.difference(&input_set).copied().collect();
        if layer_nodes.is_empty() {
            break;
        }

        required.extend(layer_nodes);
        search.extend(upstream);
    }

    required
}

pub fn feed_forward_layers(
    inputs: &[NodeKey],
    outputs: &[NodeKey],
    connections: &[ConnectionKey],
) -> (Vec<Vec<NodeKey>>, BTreeSet<NodeKey>) {
    let required = required_for_output(inputs, outputs, connections);

    let mut nodes_with_inputs = BTreeSet::new();
    for connection in connections {
        nodes_with_inputs.insert(connection.output);
    }

    let bias_neurons: BTreeSet<NodeKey> =
        required.difference(&nodes_with_inputs).copied().collect();
    let mut layers = Vec::new();
    let mut potential_input: BTreeSet<NodeKey> = inputs.iter().copied().collect();
    potential_input.extend(bias_neurons.iter().copied());

    if !bias_neurons.is_empty() {
        layers.push(bias_neurons.iter().copied().collect());
    }

    loop {
        let mut candidates = BTreeSet::new();
        for connection in connections {
            if potential_input.contains(&connection.input)
                && !potential_input.contains(&connection.output)
            {
                candidates.insert(connection.output);
            }
        }

        let mut next_layer = BTreeSet::new();
        for node in candidates {
            let incoming_required: Vec<ConnectionKey> = connections
                .iter()
                .copied()
                .filter(|connection| {
                    connection.output == node && required.contains(&connection.input)
                })
                .collect();
            if required.contains(&node)
                && incoming_required
                    .iter()
                    .all(|connection| potential_input.contains(&connection.input))
            {
                next_layer.insert(node);
            }
        }

        if next_layer.is_empty() {
            break;
        }

        layers.push(next_layer.iter().copied().collect());
        potential_input.extend(next_layer);
    }

    (layers, required)
}
