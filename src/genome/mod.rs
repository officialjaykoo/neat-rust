use std::collections::{BTreeMap, BTreeSet};

use crate::attributes::RandomSource;
use crate::config::{FitnessCriterion, GenomeConfig, InitialConnectionMode};
use crate::gene::{ConnectionKey, DefaultConnectionGene, DefaultNodeGene, NodeKey};
use crate::graph::{creates_cycle, required_for_output};
use crate::ids::GenomeId;
use crate::innovation::{InnovationTracker, MutationType};

mod error;

pub use error::GenomeError;

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultGenome {
    pub key: GenomeId,
    pub connections: BTreeMap<ConnectionKey, DefaultConnectionGene>,
    pub nodes: BTreeMap<NodeKey, DefaultNodeGene>,
    pub fitness: Option<f64>,
}

impl DefaultGenome {
    pub fn new(key: impl Into<GenomeId>) -> Self {
        Self {
            key: key.into(),
            connections: BTreeMap::new(),
            nodes: BTreeMap::new(),
            fitness: None,
        }
    }

    pub fn configure_new(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        self.connections.clear();
        self.nodes.clear();
        self.fitness = None;

        for node_key in output_keys(config) {
            self.nodes
                .insert(node_key, Self::create_node(config, node_key, rng)?);
        }

        for _ in 0..config.num_hidden {
            let node_key = next_node_key(config, &self.nodes);
            self.nodes
                .insert(node_key, Self::create_node(config, node_key, rng)?);
        }

        match &config.initial_connection.mode {
            InitialConnectionMode::Unconnected => Ok(()),
            InitialConnectionMode::FullNoDirect => self.connect_full_nodirect(config, rng),
            InitialConnectionMode::FullDirect => self.connect_full_direct(config, rng),
            InitialConnectionMode::Full => self.connect_full_nodirect(config, rng),
            InitialConnectionMode::PartialNoDirect => {
                self.connect_partial_nodirect(config, config.initial_connection.fraction, rng)
            }
            InitialConnectionMode::PartialDirect | InitialConnectionMode::Partial => {
                self.connect_partial_direct(config, config.initial_connection.fraction, rng)
            }
            InitialConnectionMode::Unsupported(value) => {
                Err(GenomeError::UnsupportedInitialConnection(value.clone()))
            }
        }
    }

    pub fn configure_new_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        self.connections.clear();
        self.nodes.clear();
        self.fitness = None;

        for node_key in output_keys(config) {
            self.nodes
                .insert(node_key, Self::create_node(config, node_key, rng)?);
        }

        for _ in 0..config.num_hidden {
            let node_key = next_node_key(config, &self.nodes);
            self.nodes
                .insert(node_key, Self::create_node(config, node_key, rng)?);
        }

        match &config.initial_connection.mode {
            InitialConnectionMode::Unconnected => Ok(()),
            InitialConnectionMode::FullNoDirect => {
                self.connect_full_nodirect_with_innovation(config, tracker, rng)
            }
            InitialConnectionMode::FullDirect => {
                self.connect_full_direct_with_innovation(config, tracker, rng)
            }
            InitialConnectionMode::Full => {
                self.connect_full_nodirect_with_innovation(config, tracker, rng)
            }
            InitialConnectionMode::PartialNoDirect => self
                .connect_partial_nodirect_with_innovation(
                    config,
                    config.initial_connection.fraction,
                    tracker,
                    rng,
                ),
            InitialConnectionMode::PartialDirect | InitialConnectionMode::Partial => self
                .connect_partial_direct_with_innovation(
                    config,
                    config.initial_connection.fraction,
                    tracker,
                    rng,
                ),
            InitialConnectionMode::Unsupported(value) => {
                Err(GenomeError::UnsupportedInitialConnection(value.clone()))
            }
        }
    }

    pub fn configure_crossover(
        &mut self,
        genome1: &DefaultGenome,
        genome2: &DefaultGenome,
        config: &GenomeConfig,
        fitness_criterion: Option<&FitnessCriterion>,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        self.connections.clear();
        self.nodes.clear();
        self.fitness = None;

        let fitness1 = genome1
            .fitness
            .ok_or(GenomeError::MissingFitness(genome1.key))?;
        let fitness2 = genome2
            .fitness
            .ok_or(GenomeError::MissingFitness(genome2.key))?;
        let genome1_is_better =
            if fitness_criterion.map(|criterion| criterion.is_min()) == Some(true) {
                fitness1 < fitness2
            } else {
                fitness1 > fitness2
            };
        let (parent1, parent2) = if genome1_is_better {
            (genome1, genome2)
        } else {
            (genome2, genome1)
        };

        if connections_have_innovations(parent1) && connections_have_innovations(parent2) {
            self.configure_crossover_by_innovation(parent1, parent2, config, rng)?;
        } else {
            self.configure_crossover_by_key(parent1, parent2, config, rng)?;
        }

        for (key, node1) in &parent1.nodes {
            if let Some(node2) = parent2.nodes.get(key) {
                self.nodes.insert(*key, node1.crossover(node2, rng)?);
            } else {
                self.nodes.insert(*key, node1.clone());
            }
        }

        Ok(())
    }

    fn configure_crossover_by_innovation(
        &mut self,
        parent1: &DefaultGenome,
        parent2: &DefaultGenome,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        let parent1_innovations = connection_innovation_map(parent1);
        let parent2_innovations = connection_innovation_map(parent2);
        let mut innovations: BTreeSet<i64> = parent1_innovations.keys().copied().collect();
        innovations.extend(parent2_innovations.keys().copied());

        for innovation in innovations {
            let gene1 = parent1_innovations.get(&innovation).copied();
            let gene2 = parent2_innovations.get(&innovation).copied();
            let new_gene = match (gene1, gene2) {
                (Some(gene1), Some(gene2)) if gene1.key == gene2.key => {
                    gene1.crossover(gene2, rng)?
                }
                (Some(gene1), _) => gene1.clone(),
                (None, Some(_)) | (None, None) => continue,
            };
            self.inherit_connection_if_allowed(new_gene, config);
        }

        Ok(())
    }

    fn configure_crossover_by_key(
        &mut self,
        parent1: &DefaultGenome,
        parent2: &DefaultGenome,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for (key, gene1) in &parent1.connections {
            let new_gene = if let Some(gene2) = parent2.connections.get(key) {
                gene1.crossover(gene2, rng)?
            } else {
                gene1.clone()
            };
            self.inherit_connection_if_allowed(new_gene, config);
        }

        Ok(())
    }

    fn inherit_connection_if_allowed(
        &mut self,
        connection: DefaultConnectionGene,
        config: &GenomeConfig,
    ) {
        if config.feed_forward {
            let existing: Vec<ConnectionKey> = self.connections.keys().copied().collect();
            if creates_cycle(&existing, connection.key) {
                return;
            }
        }
        self.connections.insert(connection.key, connection);
    }

    pub fn mutate(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        if config.single_structural_mutation {
            let div = (config.node_add_prob
                + config.node_delete_prob
                + config.conn_add_prob
                + config.conn_delete_prob)
                .max(1.0);
            let roll = rng.next_f64();
            if roll < config.node_add_prob / div {
                self.mutate_add_node(config, rng)?;
            } else if roll < (config.node_add_prob + config.node_delete_prob) / div {
                self.mutate_delete_node(config, rng)?;
            } else if roll
                < (config.node_add_prob + config.node_delete_prob + config.conn_add_prob) / div
            {
                self.mutate_add_connection(config, rng)?;
            } else if roll
                < (config.node_add_prob
                    + config.node_delete_prob
                    + config.conn_add_prob
                    + config.conn_delete_prob)
                    / div
            {
                self.mutate_delete_connection(config, rng);
            }
        } else {
            if rng.next_f64() < config.node_add_prob {
                self.mutate_add_node(config, rng)?;
            }
            if rng.next_f64() < config.node_delete_prob {
                self.mutate_delete_node(config, rng)?;
            }
            if rng.next_f64() < config.conn_add_prob {
                self.mutate_add_connection(config, rng)?;
            }
            if rng.next_f64() < config.conn_delete_prob {
                self.mutate_delete_connection(config, rng);
            }
        }

        for connection in self.connections.values_mut() {
            connection.mutate(config, rng)?;
        }
        for node in self.nodes.values_mut() {
            node.mutate(config, rng)?;
        }

        Ok(())
    }

    pub fn mutate_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        if config.single_structural_mutation {
            let div = (config.node_add_prob
                + config.node_delete_prob
                + config.conn_add_prob
                + config.conn_delete_prob)
                .max(1.0);
            let roll = rng.next_f64();
            if roll < config.node_add_prob / div {
                self.mutate_add_node_with_innovation(config, tracker, rng)?;
            } else if roll < (config.node_add_prob + config.node_delete_prob) / div {
                self.mutate_delete_node(config, rng)?;
            } else if roll
                < (config.node_add_prob + config.node_delete_prob + config.conn_add_prob) / div
            {
                self.mutate_add_connection_with_innovation(config, tracker, rng)?;
            } else if roll
                < (config.node_add_prob
                    + config.node_delete_prob
                    + config.conn_add_prob
                    + config.conn_delete_prob)
                    / div
            {
                self.mutate_delete_connection(config, rng);
            }
        } else {
            if rng.next_f64() < config.node_add_prob {
                self.mutate_add_node_with_innovation(config, tracker, rng)?;
            }
            if rng.next_f64() < config.node_delete_prob {
                self.mutate_delete_node(config, rng)?;
            }
            if rng.next_f64() < config.conn_add_prob {
                self.mutate_add_connection_with_innovation(config, tracker, rng)?;
            }
            if rng.next_f64() < config.conn_delete_prob {
                self.mutate_delete_connection(config, rng);
            }
        }

        for connection in self.connections.values_mut() {
            connection.mutate(config, rng)?;
        }
        for node in self.nodes.values_mut() {
            node.mutate(config, rng)?;
        }

        Ok(())
    }

    pub fn mutate_add_node(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Option<NodeKey>, GenomeError> {
        if self.connections.is_empty() {
            if check_structural_mutation_surer(config) {
                self.mutate_add_connection(config, rng)?;
            }
            return Ok(None);
        }

        let connection_keys: Vec<ConnectionKey> = self.connections.keys().copied().collect();
        let split_key = choose_connection_key(&connection_keys, rng, "connections")?;
        let connection_weight = self
            .connections
            .get(&split_key)
            .map(|connection| connection.weight)
            .unwrap_or(0.0);
        let new_node_id = next_node_key(config, &self.nodes);
        let mut node = Self::create_node(config, new_node_id, rng)?;
        node.bias = 0.0;
        self.nodes.insert(new_node_id, node);

        if let Some(connection) = self.connections.get_mut(&split_key) {
            connection.enabled = false;
        }

        let input_id = split_key.input;
        let output_id = split_key.output;
        self.add_connection(config, input_id, new_node_id, 1.0, true, rng)?;
        self.add_connection(config, new_node_id, output_id, connection_weight, true, rng)?;

        Ok(Some(new_node_id))
    }

    pub fn mutate_delete_node(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Option<NodeKey>, GenomeError> {
        let outputs: BTreeSet<NodeKey> = output_keys(config).into_iter().collect();
        let available_nodes: Vec<NodeKey> = self
            .nodes
            .keys()
            .copied()
            .filter(|node_key| !outputs.contains(node_key))
            .collect();
        if available_nodes.is_empty() {
            return Ok(None);
        }

        let deleted = choose_node_key(&available_nodes, rng, "available_nodes")?;
        let connections_to_delete: Vec<ConnectionKey> = self
            .connections
            .keys()
            .copied()
            .filter(|key| key.input == deleted || key.output == deleted)
            .collect();
        for key in connections_to_delete {
            self.connections.remove(&key);
        }
        self.nodes.remove(&deleted);
        self.prune_dangling_nodes(config);

        Ok(Some(deleted))
    }

    pub fn mutate_add_node_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<Option<NodeKey>, GenomeError> {
        if self.connections.is_empty() {
            if check_structural_mutation_surer(config) {
                self.mutate_add_connection_with_innovation(config, tracker, rng)?;
            }
            return Ok(None);
        }

        let connection_keys: Vec<ConnectionKey> = self.connections.keys().copied().collect();
        let split_key = choose_connection_key(&connection_keys, rng, "connections")?;
        let connection_weight = self
            .connections
            .get(&split_key)
            .map(|connection| connection.weight)
            .unwrap_or(0.0);
        let new_node_id = next_node_key(config, &self.nodes);
        let mut node = Self::create_node(config, new_node_id, rng)?;
        node.bias = 0.0;
        self.nodes.insert(new_node_id, node);

        if let Some(connection) = self.connections.get_mut(&split_key) {
            connection.enabled = false;
        }

        let input_id = split_key.input;
        let output_id = split_key.output;
        let in_innovation =
            tracker.get_innovation_number(input_id, new_node_id, MutationType::AddNodeIn);
        let out_innovation =
            tracker.get_innovation_number(new_node_id, output_id, MutationType::AddNodeOut);
        self.add_connection_with_innovation(
            config,
            input_id,
            new_node_id,
            1.0,
            true,
            in_innovation,
            rng,
        )?;
        self.add_connection_with_innovation(
            config,
            new_node_id,
            output_id,
            connection_weight,
            true,
            out_innovation,
            rng,
        )?;

        Ok(Some(new_node_id))
    }

    pub fn mutate_add_connection(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<Option<ConnectionKey>, GenomeError> {
        let possible_outputs: Vec<NodeKey> = self.nodes.keys().copied().collect();
        let output_node = choose_node_key(&possible_outputs, rng, "possible_outputs")?;

        let outputs: BTreeSet<NodeKey> = output_keys(config).into_iter().collect();
        let mut possible_inputs: BTreeSet<NodeKey> = input_keys(config).into_iter().collect();
        for node_key in self.nodes.keys().copied() {
            if !outputs.contains(&node_key) {
                possible_inputs.insert(node_key);
            }
        }
        let possible_inputs: Vec<NodeKey> = possible_inputs.into_iter().collect();
        let input_node = choose_node_key(&possible_inputs, rng, "possible_inputs")?;
        let key = ConnectionKey::new(input_node, output_node);

        if let Some(connection) = self.connections.get_mut(&key) {
            if check_structural_mutation_surer(config) {
                connection.enabled = true;
            }
            return Ok(None);
        }

        if outputs.contains(&input_node) && outputs.contains(&output_node) {
            return Ok(None);
        }

        if config.feed_forward {
            let existing: Vec<ConnectionKey> = self.connections.keys().copied().collect();
            if creates_cycle(&existing, key) {
                return Ok(None);
            }
        }

        let connection = Self::create_connection(config, input_node, output_node, rng)?;
        let connection_key = connection.key;
        self.connections.insert(connection_key, connection);
        Ok(Some(connection_key))
    }

    pub fn mutate_add_connection_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<Option<ConnectionKey>, GenomeError> {
        let possible_outputs: Vec<NodeKey> = self.nodes.keys().copied().collect();
        let output_node = choose_node_key(&possible_outputs, rng, "possible_outputs")?;

        let outputs: BTreeSet<NodeKey> = output_keys(config).into_iter().collect();
        let mut possible_inputs: BTreeSet<NodeKey> = input_keys(config).into_iter().collect();
        for node_key in self.nodes.keys().copied() {
            if !outputs.contains(&node_key) {
                possible_inputs.insert(node_key);
            }
        }
        let possible_inputs: Vec<NodeKey> = possible_inputs.into_iter().collect();
        let input_node = choose_node_key(&possible_inputs, rng, "possible_inputs")?;
        let key = ConnectionKey::new(input_node, output_node);

        if let Some(connection) = self.connections.get_mut(&key) {
            if check_structural_mutation_surer(config) {
                connection.enabled = true;
            }
            return Ok(None);
        }

        if outputs.contains(&input_node) && outputs.contains(&output_node) {
            return Ok(None);
        }

        if config.feed_forward {
            let existing: Vec<ConnectionKey> = self.connections.keys().copied().collect();
            if creates_cycle(&existing, key) {
                return Ok(None);
            }
        }

        let innovation =
            tracker.get_innovation_number(input_node, output_node, MutationType::AddConnection);
        let connection = Self::create_connection_with_innovation(
            config,
            input_node,
            output_node,
            innovation,
            rng,
        )?;
        let connection_key = connection.key;
        self.connections.insert(connection_key, connection);
        Ok(Some(connection_key))
    }

    pub fn mutate_delete_connection(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Option<ConnectionKey> {
        if self.connections.is_empty() {
            return None;
        }

        let connection_keys: Vec<ConnectionKey> = self.connections.keys().copied().collect();
        let key = choose_connection_key(&connection_keys, rng, "connections").ok()?;
        self.connections.remove(&key);
        self.prune_dangling_nodes(config);
        Some(key)
    }

    pub fn add_connection(
        &mut self,
        config: &GenomeConfig,
        input_id: NodeKey,
        output_id: NodeKey,
        weight: f64,
        enabled: bool,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        if output_id < 0 {
            return Err(GenomeError::InvalidConnection(ConnectionKey::new(
                input_id, output_id,
            )));
        }

        let mut connection = Self::create_connection(config, input_id, output_id, rng)?;
        connection.weight = weight;
        connection.enabled = enabled;
        self.connections.insert(connection.key, connection);
        Ok(())
    }

    pub fn add_connection_with_innovation(
        &mut self,
        config: &GenomeConfig,
        input_id: NodeKey,
        output_id: NodeKey,
        weight: f64,
        enabled: bool,
        innovation: i64,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        if output_id < 0 {
            return Err(GenomeError::InvalidConnection(ConnectionKey::new(
                input_id, output_id,
            )));
        }

        let mut connection =
            Self::create_connection_with_innovation(config, input_id, output_id, innovation, rng)?;
        connection.weight = weight;
        connection.enabled = enabled;
        self.connections.insert(connection.key, connection);
        Ok(())
    }

    pub fn prune_dangling_nodes(&mut self, config: &GenomeConfig) {
        let outputs: BTreeSet<NodeKey> = output_keys(config).into_iter().collect();
        let inputs = input_keys(config);
        let enabled_connections: Vec<ConnectionKey> = self
            .connections
            .values()
            .filter(|connection| connection.enabled)
            .map(|connection| connection.key)
            .collect();
        let required = required_for_output(&inputs, &output_keys(config), &enabled_connections);

        let nodes_to_remove: Vec<NodeKey> = self
            .nodes
            .keys()
            .copied()
            .filter(|node_key| !outputs.contains(node_key) && !required.contains(node_key))
            .collect();
        for node_key in nodes_to_remove {
            self.nodes.remove(&node_key);
            let connections_to_remove: Vec<ConnectionKey> = self
                .connections
                .keys()
                .copied()
                .filter(|key| key.input == node_key || key.output == node_key)
                .collect();
            for connection_key in connections_to_remove {
                self.connections.remove(&connection_key);
            }
        }
    }

    pub fn create_node(
        config: &GenomeConfig,
        node_id: NodeKey,
        rng: &mut impl RandomSource,
    ) -> Result<DefaultNodeGene, GenomeError> {
        Ok(DefaultNodeGene::initialized(node_id, config, rng)?)
    }

    pub fn create_connection(
        config: &GenomeConfig,
        input_id: NodeKey,
        output_id: NodeKey,
        rng: &mut impl RandomSource,
    ) -> Result<DefaultConnectionGene, GenomeError> {
        Ok(DefaultConnectionGene::initialized(
            ConnectionKey::new(input_id, output_id),
            config,
            rng,
        )?)
    }

    pub fn create_connection_with_innovation(
        config: &GenomeConfig,
        input_id: NodeKey,
        output_id: NodeKey,
        innovation: i64,
        rng: &mut impl RandomSource,
    ) -> Result<DefaultConnectionGene, GenomeError> {
        Ok(DefaultConnectionGene::initialized_with_innovation(
            ConnectionKey::new(input_id, output_id),
            innovation,
            config,
            rng,
        )?)
    }

    pub fn compute_full_connections(
        &self,
        config: &GenomeConfig,
        direct: bool,
    ) -> Vec<ConnectionKey> {
        let output_keys = output_keys(config);
        let hidden: Vec<NodeKey> = self
            .nodes
            .keys()
            .copied()
            .filter(|node_key| !output_keys.contains(node_key))
            .collect();
        let output: Vec<NodeKey> = self
            .nodes
            .keys()
            .copied()
            .filter(|node_key| output_keys.contains(node_key))
            .collect();
        let input = input_keys(config);
        let mut connections = Vec::new();

        if !hidden.is_empty() {
            for input_id in &input {
                for hidden_id in &hidden {
                    connections.push(ConnectionKey::new(*input_id, *hidden_id));
                }
            }
            for hidden_id in &hidden {
                for output_id in &output {
                    connections.push(ConnectionKey::new(*hidden_id, *output_id));
                }
            }
        }

        if direct || hidden.is_empty() {
            for input_id in &input {
                for output_id in &output {
                    connections.push(ConnectionKey::new(*input_id, *output_id));
                }
            }
        }

        if !config.feed_forward {
            for node_id in self.nodes.keys() {
                connections.push(ConnectionKey::new(*node_id, *node_id));
            }
        }

        connections
    }

    pub fn connect_full_nodirect(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in self.compute_full_connections(config, false) {
            let connection = Self::create_connection(config, key.input, key.output, rng)?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_full_direct(
        &mut self,
        config: &GenomeConfig,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in self.compute_full_connections(config, true) {
            let connection = Self::create_connection(config, key.input, key.output, rng)?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_partial_nodirect(
        &mut self,
        config: &GenomeConfig,
        fraction: f64,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in
            choose_partial_connections(self.compute_full_connections(config, false), fraction, rng)
        {
            let connection = Self::create_connection(config, key.input, key.output, rng)?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_partial_direct(
        &mut self,
        config: &GenomeConfig,
        fraction: f64,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in
            choose_partial_connections(self.compute_full_connections(config, true), fraction, rng)
        {
            let connection = Self::create_connection(config, key.input, key.output, rng)?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_full_nodirect_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in self.compute_full_connections(config, false) {
            let innovation = tracker.get_innovation_number(
                key.input,
                key.output,
                MutationType::InitialConnection,
            );
            let connection = Self::create_connection_with_innovation(
                config, key.input, key.output, innovation, rng,
            )?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_full_direct_with_innovation(
        &mut self,
        config: &GenomeConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in self.compute_full_connections(config, true) {
            let innovation = tracker.get_innovation_number(
                key.input,
                key.output,
                MutationType::InitialConnection,
            );
            let connection = Self::create_connection_with_innovation(
                config, key.input, key.output, innovation, rng,
            )?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_partial_nodirect_with_innovation(
        &mut self,
        config: &GenomeConfig,
        fraction: f64,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in
            choose_partial_connections(self.compute_full_connections(config, false), fraction, rng)
        {
            let innovation = tracker.get_innovation_number(
                key.input,
                key.output,
                MutationType::InitialConnection,
            );
            let connection = Self::create_connection_with_innovation(
                config, key.input, key.output, innovation, rng,
            )?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn connect_partial_direct_with_innovation(
        &mut self,
        config: &GenomeConfig,
        fraction: f64,
        tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        for key in
            choose_partial_connections(self.compute_full_connections(config, true), fraction, rng)
        {
            let innovation = tracker.get_innovation_number(
                key.input,
                key.output,
                MutationType::InitialConnection,
            );
            let connection = Self::create_connection_with_innovation(
                config, key.input, key.output, innovation, rng,
            )?;
            self.connections.insert(connection.key, connection);
        }
        Ok(())
    }

    pub fn size(&self) -> (usize, usize) {
        let enabled_connections = self
            .connections
            .values()
            .filter(|connection| connection.enabled)
            .count();
        (self.nodes.len(), enabled_connections)
    }

    pub fn distance(
        &self,
        other: &DefaultGenome,
        config: &GenomeConfig,
    ) -> Result<f64, GenomeError> {
        let mut node_distance = 0.0;
        if config.compatibility_include_node_genes
            && (!self.nodes.is_empty() || !other.nodes.is_empty())
        {
            let mut disjoint_nodes = 0;
            for key in other.nodes.keys() {
                if !self.nodes.contains_key(key) {
                    disjoint_nodes += 1;
                }
            }
            for (key, node) in &self.nodes {
                if let Some(other_node) = other.nodes.get(key) {
                    node_distance += node.distance(other_node, config)?;
                } else {
                    disjoint_nodes += 1;
                }
            }
            let max_nodes = self.nodes.len().max(other.nodes.len()).max(1);
            node_distance = (node_distance
                + config.compatibility_disjoint_coefficient * disjoint_nodes as f64)
                / max_nodes as f64;
        }

        let mut connection_distance = 0.0;
        if !self.connections.is_empty() || !other.connections.is_empty() {
            if connections_have_innovations(self) && connections_have_innovations(other) {
                let self_by_innovation = connection_innovation_map(self);
                let other_by_innovation = connection_innovation_map(other);
                let mut innovations: BTreeSet<i64> = self_by_innovation.keys().copied().collect();
                innovations.extend(other_by_innovation.keys().copied());
                let self_max_innovation = self_by_innovation.keys().copied().max().unwrap_or(0);
                let other_max_innovation = other_by_innovation.keys().copied().max().unwrap_or(0);
                let excess_coefficient = compatibility_excess_coefficient(config);
                let mut disjoint_connections = 0;
                let mut excess_connections = 0;
                for innovation in innovations {
                    match (
                        self_by_innovation.get(&innovation),
                        other_by_innovation.get(&innovation),
                    ) {
                        (Some(left), Some(right)) if left.key == right.key => {
                            connection_distance += left.distance(right, config)?;
                        }
                        (Some(_), Some(_)) => {
                            disjoint_connections += 2;
                        }
                        (Some(_), None) => {
                            if innovation > other_max_innovation {
                                excess_connections += 1;
                            } else {
                                disjoint_connections += 1;
                            }
                        }
                        (None, Some(_)) => {
                            if innovation > self_max_innovation {
                                excess_connections += 1;
                            } else {
                                disjoint_connections += 1;
                            }
                        }
                        (None, None) => {}
                    }
                }
                let max_connections = self_by_innovation
                    .len()
                    .max(other_by_innovation.len())
                    .max(1);
                connection_distance = (connection_distance
                    + config.compatibility_disjoint_coefficient * disjoint_connections as f64)
                    + excess_coefficient * excess_connections as f64;
                connection_distance /= max_connections as f64;
            } else {
                let mut disjoint_connections = 0;
                for key in other.connections.keys() {
                    if !self.connections.contains_key(key) {
                        disjoint_connections += 1;
                    }
                }
                for (key, connection) in &self.connections {
                    if let Some(other_connection) = other.connections.get(key) {
                        connection_distance += connection.distance(other_connection, config)?;
                    } else {
                        disjoint_connections += 1;
                    }
                }
                let max_connections = self.connections.len().max(other.connections.len()).max(1);
                connection_distance = (connection_distance
                    + config.compatibility_disjoint_coefficient * disjoint_connections as f64)
                    / max_connections as f64;
            }
        }

        Ok(node_distance + connection_distance)
    }
}

fn compatibility_excess_coefficient(config: &GenomeConfig) -> f64 {
    config
        .compatibility_excess_coefficient
        .resolve(config.compatibility_disjoint_coefficient)
}

pub fn input_keys(config: &GenomeConfig) -> Vec<NodeKey> {
    (1..=config.num_inputs).map(|i| -(i as NodeKey)).collect()
}

pub fn output_keys(config: &GenomeConfig) -> Vec<NodeKey> {
    (0..config.num_outputs).map(|i| i as NodeKey).collect()
}

fn next_node_key(config: &GenomeConfig, nodes: &BTreeMap<NodeKey, DefaultNodeGene>) -> NodeKey {
    nodes
        .keys()
        .copied()
        .max()
        .map(|key| key + 1)
        .unwrap_or(config.num_outputs as NodeKey)
}

fn choose_partial_connections(
    mut candidates: Vec<ConnectionKey>,
    fraction: f64,
    rng: &mut impl RandomSource,
) -> Vec<ConnectionKey> {
    if candidates.is_empty() {
        return candidates;
    }

    let count = ((candidates.len() as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
    if count >= candidates.len() {
        return candidates;
    }
    if count == 0 {
        return Vec::new();
    }

    for idx in 0..count {
        let remaining = candidates.len() - idx;
        let offset = rng.next_index(remaining).unwrap_or(0);
        candidates.swap(idx, idx + offset);
    }
    candidates.truncate(count);
    candidates
}

fn check_structural_mutation_surer(config: &GenomeConfig) -> bool {
    config
        .structural_mutation_surer
        .is_enabled(config.single_structural_mutation)
}

fn choose_node_key(
    values: &[NodeKey],
    rng: &mut impl RandomSource,
    name: &'static str,
) -> Result<NodeKey, GenomeError> {
    let index = rng
        .next_index(values.len())
        .ok_or(GenomeError::EmptyChoice(name))?;
    Ok(values[index])
}

fn choose_connection_key(
    values: &[ConnectionKey],
    rng: &mut impl RandomSource,
    name: &'static str,
) -> Result<ConnectionKey, GenomeError> {
    let index = rng
        .next_index(values.len())
        .ok_or(GenomeError::EmptyChoice(name))?;
    Ok(values[index])
}

fn connections_have_innovations(genome: &DefaultGenome) -> bool {
    !genome.connections.is_empty()
        && genome
            .connections
            .values()
            .all(|connection| connection.innovation.is_some())
}

fn connection_innovation_map(genome: &DefaultGenome) -> BTreeMap<i64, &DefaultConnectionGene> {
    genome
        .connections
        .values()
        .filter_map(|connection| {
            connection
                .innovation
                .map(|innovation| (innovation, connection))
        })
        .collect()
}
