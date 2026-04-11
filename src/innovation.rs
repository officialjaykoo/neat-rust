use std::collections::BTreeMap;

use crate::gene::NodeKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MutationType {
    AddConnection,
    AddNodeIn,
    AddNodeOut,
    InitialConnection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnovationTracker {
    global_counter: i64,
    generation_innovations: BTreeMap<InnovationKey, i64>,
}

type InnovationKey = (NodeKey, NodeKey, MutationType);

impl InnovationTracker {
    pub fn new() -> Self {
        Self::with_start_number(0)
    }

    pub fn with_start_number(start_number: i64) -> Self {
        Self {
            global_counter: start_number,
            generation_innovations: BTreeMap::new(),
        }
    }

    pub fn get_innovation_number(
        &mut self,
        input_node: NodeKey,
        output_node: NodeKey,
        mutation_type: MutationType,
    ) -> i64 {
        let key = (input_node, output_node, mutation_type);
        if let Some(innovation) = self.generation_innovations.get(&key) {
            return *innovation;
        }

        self.global_counter += 1;
        let innovation = self.global_counter;
        self.generation_innovations.insert(key, innovation);
        innovation
    }

    pub fn reset_generation(&mut self) {
        self.generation_innovations.clear();
    }

    pub fn current_innovation_number(&self) -> i64 {
        self.global_counter
    }

    pub fn tracked_generation_innovations(&self) -> usize {
        self.generation_innovations.len()
    }
}

impl Default for InnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}
