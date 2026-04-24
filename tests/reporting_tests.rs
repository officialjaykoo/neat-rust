use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use neat_rust::{Config, GenomeId, Population, Reporter, ReporterSet, SpeciesId};

fn repo_path(relative: &str) -> PathBuf {
    let relative = relative.strip_prefix("scripts/configs/").unwrap_or(relative);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("configs")
        .join(relative)
}

#[test]
fn reporter_set_dispatches_core_events_to_all_reporters() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    let population = Population::new(config, 24).expect("population should initialize");
    let best = population
        .population
        .values()
        .next()
        .expect("population should have genomes")
        .clone();
    let first_species = population
        .species
        .species
        .values()
        .next()
        .expect("population should have species");

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut reporters = ReporterSet::new();
    reporters.add(Box::new(EventReporter {
        events: events.clone(),
    }));
    reporters.add(Box::new(EventReporter {
        events: events.clone(),
    }));

    reporters.start_generation(3);
    reporters.post_evaluate(
        &population.config,
        &population.population,
        &population.species,
        &best,
    );
    reporters.post_reproduction(
        &population.config,
        &population.population,
        &population.species,
    );
    reporters.complete_extinction();
    reporters.found_solution(&population.config, 3, &best);
    reporters.species_stagnant(first_species.key, first_species);
    reporters.info("hello");

    let events = events.borrow();
    let counts = [
        ("start:3", 2usize),
        ("post_evaluate", 2),
        ("post_reproduction", 2),
        ("complete_extinction", 2),
        ("found_solution:3", 2),
        ("species_stagnant", 2),
        ("info:hello", 2),
    ];

    for (event, expected) in counts {
        assert_eq!(
            events.iter().filter(|seen| seen.as_str() == event).count(),
            expected
        );
    }
}

struct EventReporter {
    events: Rc<RefCell<Vec<String>>>,
}

impl Reporter for EventReporter {
    fn start_generation(&mut self, generation: usize) {
        self.events.borrow_mut().push(format!("start:{generation}"));
    }

    fn post_evaluate(
        &mut self,
        _config: &Config,
        _population: &std::collections::BTreeMap<GenomeId, neat_rust::DefaultGenome>,
        _species: &neat_rust::SpeciesSet,
        _best_genome: &neat_rust::DefaultGenome,
    ) {
        self.events.borrow_mut().push("post_evaluate".to_string());
    }

    fn post_reproduction(
        &mut self,
        _config: &Config,
        _population: &std::collections::BTreeMap<GenomeId, neat_rust::DefaultGenome>,
        _species: &neat_rust::SpeciesSet,
    ) {
        self.events
            .borrow_mut()
            .push("post_reproduction".to_string());
    }

    fn complete_extinction(&mut self) {
        self.events
            .borrow_mut()
            .push("complete_extinction".to_string());
    }

    fn found_solution(
        &mut self,
        _config: &Config,
        generation: usize,
        _best: &neat_rust::DefaultGenome,
    ) {
        self.events
            .borrow_mut()
            .push(format!("found_solution:{generation}"));
    }

    fn species_stagnant(&mut self, _species_id: SpeciesId, _species: &neat_rust::Species) {
        self.events
            .borrow_mut()
            .push("species_stagnant".to_string());
    }

    fn info(&mut self, message: &str) {
        self.events.borrow_mut().push(format!("info:{message}"));
    }
}
