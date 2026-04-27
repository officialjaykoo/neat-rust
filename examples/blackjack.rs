use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use neat_rust::{
    algorithm::{
        BootstrapStrategy, DefaultGenome, FitnessError, FitnessEvaluator, GenomeId, Population,
        RandomSource, XorShiftRng,
    },
    io::{export_neat_genome_json, new_rust_checkpoint_sink, Config},
    network::RecurrentNetwork,
};

const INPUT_COUNT: usize = 14;
const OUTPUT_COUNT: usize = 5;
const LINEAR_PARAM_COUNT: usize = OUTPUT_COUNT * (INPUT_COUNT + 1);
const DEFAULT_BASE_SEED: u64 = 505;
const DECKS: usize = 6;
const PENETRATION_CARDS: usize = 52;
const MAX_ROUNDS_PER_SHOE: usize = 90;
const MAX_PLAYER_CARDS: usize = 9;
const MAX_SPLITS: usize = 3;
const DEFAULT_REPORT_ROUNDS: usize = 1_000;
const TRAIN_SHOES: [u64; 10] = [13, 29, 43, 61, 83, 107, 131, 167, 193, 227];
const REPORT_SEEDS: [u64; 12] = [251, 277, 307, 337, 367, 397, 431, 463, 491, 521, 557, 587];

#[derive(Debug, Clone)]
struct Shoe {
    cards: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Hand {
    cards: Vec<u8>,
    bet: f64,
    split: bool,
    doubled: bool,
    surrendered: bool,
}

#[derive(Debug, Clone, Copy)]
struct DecisionState {
    total: u8,
    dealer_upcard: u8,
    soft: bool,
    pair_rank: u8,
    card_count: u8,
    can_hit: bool,
    can_double: bool,
    can_split: bool,
    can_surrender: bool,
    last_action: Action,
    phase: Phase,
    exposed_card: u8,
    shoe_progress: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ShoeReport {
    fitness: f64,
    profit: f64,
    wins: usize,
    losses: usize,
    pushes: usize,
    busts: usize,
    blackjacks: usize,
    doubled: usize,
    split: usize,
    surrendered: usize,
    rounds: usize,
    hands: usize,
}

struct BlackjackEvaluator;

trait BlackjackPolicy {
    fn reset(&mut self);
    fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, FitnessError>;
}

impl BlackjackPolicy for RecurrentNetwork {
    fn reset(&mut self) {
        RecurrentNetwork::reset(self);
    }

    fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, FitnessError> {
        RecurrentNetwork::activate(self, inputs).map_err(|err| FitnessError::new(err.to_string()))
    }
}

#[derive(Debug, Clone)]
struct LinearBlackjackPolicy {
    params: Vec<f64>,
}

impl LinearBlackjackPolicy {
    fn new(params: &[f64]) -> Self {
        Self {
            params: params.to_vec(),
        }
    }
}

impl BlackjackPolicy for LinearBlackjackPolicy {
    fn reset(&mut self) {}

    fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, FitnessError> {
        let mut outputs = vec![0.0; OUTPUT_COUNT];
        for (output_idx, output) in outputs.iter_mut().enumerate() {
            let base = output_idx * (INPUT_COUNT + 1);
            let mut sum = self.params.get(base + INPUT_COUNT).copied().unwrap_or(0.0);
            for input_idx in 0..INPUT_COUNT {
                sum += inputs.get(input_idx).copied().unwrap_or(0.0)
                    * self.params.get(base + input_idx).copied().unwrap_or(0.0);
            }
            *output = sum.tanh();
        }
        Ok(outputs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlackjackConfigProfile {
    Plain,
    NodeGru,
    Hebbian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Action {
    #[default]
    None,
    Stand,
    Hit,
    Double,
    Split,
    Surrender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Observe,
    Decide,
    Reveal,
}

impl BlackjackConfigProfile {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("plain").to_ascii_lowercase().as_str() {
            "plain" | "base" | "no-gru" | "no_gru" => Ok(Self::Plain),
            "gru" | "node-gru" | "node_gru" | "nodegru" => Ok(Self::NodeGru),
            "hebbian" | "node-hebbian" | "node_hebbian" => Ok(Self::Hebbian),
            other => Err(format!(
                "unknown blackjack config profile {other:?}; use plain, node-gru, or hebbian"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::NodeGru => "node-gru",
            Self::Hebbian => "hebbian",
        }
    }

    fn config_path(self) -> PathBuf {
        PathBuf::from("examples").join(match self {
            Self::Plain => "blackjack_plain_config.toml",
            Self::NodeGru => "blackjack_node_gru_config.toml",
            Self::Hebbian => "blackjack_hebbian_config.toml",
        })
    }
}

impl FitnessEvaluator for BlackjackEvaluator {
    fn evaluate_genome(
        &mut self,
        genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        let mut total = 0.0;
        for seed in TRAIN_SHOES {
            let mut network = RecurrentNetwork::create(genome, &config.genome)
                .map_err(|err| FitnessError::new(err.to_string()))?;
            total += run_shoe(genome_id, seed, &mut network)?.fitness;
        }
        Ok(total / TRAIN_SHOES.len() as f64)
    }
}

impl Shoe {
    fn shuffled(seed: u64) -> Self {
        let mut cards = Vec::with_capacity(DECKS * 52);
        for _ in 0..DECKS {
            for rank in 1..=13 {
                let value = if rank > 10 { 10 } else { rank as u8 };
                for _ in 0..4 {
                    cards.push(value);
                }
            }
        }

        let mut rng = XorShiftRng::seed_from_u64(seed);
        for index in (1..cards.len()).rev() {
            let swap_index = rng.next_index(index + 1).unwrap_or(0);
            cards.swap(index, swap_index);
        }
        Self { cards }
    }

    fn draw(&mut self) -> Option<u8> {
        self.cards.pop()
    }

    fn progress(&self) -> f64 {
        1.0 - (self.cards.len() as f64 / (DECKS * 52) as f64)
    }

    fn should_continue(&self) -> bool {
        self.cards.len() > PENETRATION_CARDS
    }
}

impl Hand {
    fn new() -> Self {
        Self {
            cards: Vec::with_capacity(MAX_PLAYER_CARDS),
            bet: 1.0,
            split: false,
            doubled: false,
            surrendered: false,
        }
    }

    fn add(&mut self, card: u8) {
        self.cards.push(card);
    }

    fn value(&self) -> u8 {
        let mut total = 0;
        let mut aces = 0;
        for card in &self.cards {
            if *card == 1 {
                aces += 1;
                total += 11;
            } else {
                total += *card;
            }
        }
        while total > 21 && aces > 0 {
            total -= 10;
            aces -= 1;
        }
        total
    }

    fn is_soft(&self) -> bool {
        self.cards.iter().filter(|card| **card == 1).count() > 0
            && self
                .cards
                .iter()
                .map(|card| if *card == 1 { 11 } else { *card })
                .sum::<u8>()
                <= 21
    }

    fn is_blackjack(&self) -> bool {
        !self.split && self.cards.len() == 2 && self.value() == 21
    }

    fn is_bust(&self) -> bool {
        self.value() > 21
    }

    fn pair_rank(&self) -> u8 {
        if self.cards.len() == 2 && card_score(self.cards[0]) == card_score(self.cards[1]) {
            card_score(self.cards[0])
        } else {
            0
        }
    }

    fn split_off_pair(&mut self) -> Self {
        let card = self.cards.pop().expect("split requires a pair");
        self.split = true;
        Self {
            cards: vec![card],
            bet: self.bet,
            split: true,
            doubled: false,
            surrendered: false,
        }
    }
}

impl DecisionState {
    fn inputs(self) -> [f64; 14] {
        [
            ((self.total as f64 / 21.0) * 2.0 - 1.0).clamp(-1.0, 1.0),
            ((card_score(self.dealer_upcard) as f64 / 11.0) * 2.0 - 1.0).clamp(-1.0, 1.0),
            if self.soft { 1.0 } else { -1.0 },
            if self.pair_rank == 0 {
                -1.0
            } else {
                ((self.pair_rank as f64 / 11.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            },
            ((self.card_count as f64 / MAX_PLAYER_CARDS as f64) * 2.0 - 1.0).clamp(-1.0, 1.0),
            bool_input(self.can_hit),
            bool_input(self.can_double),
            bool_input(self.can_split),
            bool_input(self.can_surrender),
            self.last_action.input_value(),
            self.phase.input_value(),
            if self.exposed_card == 0 {
                -1.0
            } else {
                ((card_score(self.exposed_card) as f64 / 11.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            },
            (self.shoe_progress * 2.0 - 1.0).clamp(-1.0, 1.0),
            1.0,
        ]
    }
}

impl Action {
    fn from_outputs(outputs: &[f64], legal: &[Self]) -> Self {
        let mut best = legal.first().copied().unwrap_or(Self::Stand);
        let mut best_value = f64::NEG_INFINITY;
        for action in legal {
            let index = action.output_index();
            let value = outputs.get(index).copied().unwrap_or(f64::NEG_INFINITY);
            if value > best_value {
                best_value = value;
                best = *action;
            }
        }
        best
    }

    fn output_index(self) -> usize {
        match self {
            Self::Stand | Self::None => 0,
            Self::Hit => 1,
            Self::Double => 2,
            Self::Split => 3,
            Self::Surrender => 4,
        }
    }

    fn input_value(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Stand => -0.75,
            Self::Hit => -0.25,
            Self::Double => 0.25,
            Self::Split => 0.65,
            Self::Surrender => 1.0,
        }
    }
}

impl Phase {
    fn input_value(self) -> f64 {
        match self {
            Self::Observe => -1.0,
            Self::Decide => 0.0,
            Self::Reveal => 1.0,
        }
    }
}

fn run_shoe(
    genome_id: GenomeId,
    seed: u64,
    network: &mut dyn BlackjackPolicy,
) -> Result<ShoeReport, FitnessError> {
    network.reset();
    let genome_seed = genome_id.raw().max(0) as u64;
    let mut shoe = Shoe::shuffled(seed ^ (genome_seed << 32));
    let mut report = ShoeReport::default();
    let mut last_action = Action::None;

    while shoe.should_continue() && report.rounds < MAX_ROUNDS_PER_SHOE {
        let round_report = play_round(&mut shoe, network, &mut last_action)?;
        report.absorb(round_report);
    }

    let resolved_bonus =
        report.wins as f64 * 0.06 + report.pushes as f64 * 0.02 + report.blackjacks as f64 * 0.40;
    let action_bonus = report.doubled as f64 * 0.02 + report.split as f64 * 0.02;
    let risk_penalty = report.busts as f64 * 0.04 + report.surrendered as f64 * 0.02;
    report.fitness =
        (260.0 + (report.profit * 22.0) + resolved_bonus + action_bonus - risk_penalty).max(0.0);
    Ok(report)
}

fn play_round(
    shoe: &mut Shoe,
    network: &mut dyn BlackjackPolicy,
    last_action: &mut Action,
) -> Result<ShoeReport, FitnessError> {
    let mut report = ShoeReport {
        rounds: 1,
        ..ShoeReport::default()
    };
    let mut player_hands = vec![Hand::new()];
    let mut dealer = Hand::new();

    deal_public_card(shoe, network, &mut player_hands[0], 0, Phase::Observe)?;
    deal_public_card(shoe, network, &mut dealer, 0, Phase::Observe)?;
    deal_public_card(
        shoe,
        network,
        &mut player_hands[0],
        dealer.cards[0],
        Phase::Observe,
    )?;
    dealer.add(draw_or_err(shoe)?);

    let dealer_upcard = dealer.cards[0];
    if dealer_may_peek(dealer_upcard) && dealer.is_blackjack() {
        observe_hand(
            network,
            &dealer,
            dealer_upcard,
            *last_action,
            Phase::Reveal,
            shoe,
        )?;
        settle_all(&player_hands, &dealer, &mut report);
        return Ok(report);
    }

    if player_hands[0].is_blackjack() {
        observe_hand(
            network,
            &dealer,
            dealer_upcard,
            *last_action,
            Phase::Reveal,
            shoe,
        )?;
        settle_all(&player_hands, &dealer, &mut report);
        return Ok(report);
    }

    let mut index = 0;
    while index < player_hands.len() {
        play_player_hand(
            shoe,
            network,
            &mut player_hands,
            index,
            dealer_upcard,
            last_action,
        )?;
        index += 1;
    }

    if player_hands
        .iter()
        .any(|hand| !hand.is_bust() && !hand.surrendered)
    {
        observe_hand(
            network,
            &dealer,
            dealer_upcard,
            *last_action,
            Phase::Reveal,
            shoe,
        )?;
        while dealer_should_hit(&dealer) {
            deal_public_card(shoe, network, &mut dealer, dealer_upcard, Phase::Reveal)?;
        }
    } else {
        observe_hand(
            network,
            &dealer,
            dealer_upcard,
            *last_action,
            Phase::Reveal,
            shoe,
        )?;
    }

    settle_all(&player_hands, &dealer, &mut report);
    Ok(report)
}

fn play_player_hand(
    shoe: &mut Shoe,
    network: &mut dyn BlackjackPolicy,
    hands: &mut Vec<Hand>,
    index: usize,
    dealer_upcard: u8,
    last_action: &mut Action,
) -> Result<(), FitnessError> {
    if hands[index].cards.len() == 1 {
        deal_public_card(
            shoe,
            network,
            &mut hands[index],
            dealer_upcard,
            Phase::Observe,
        )?;
    }

    if hands[index].split && hands[index].cards[0] == 1 {
        return Ok(());
    }

    loop {
        if hands[index].is_bust() || hands[index].cards.len() >= MAX_PLAYER_CARDS {
            return Ok(());
        }

        let legal = legal_actions(&hands[index], hands.len());
        let state = state_for(
            &hands[index],
            hands.len(),
            dealer_upcard,
            *last_action,
            Phase::Decide,
            0,
            shoe,
        );
        let outputs = network.activate(&state.inputs())?;
        let action = Action::from_outputs(&outputs, &legal);
        *last_action = action;

        match action {
            Action::Stand => return Ok(()),
            Action::Hit => {
                deal_public_card(
                    shoe,
                    network,
                    &mut hands[index],
                    dealer_upcard,
                    Phase::Observe,
                )?;
            }
            Action::Double => {
                hands[index].doubled = true;
                hands[index].bet *= 2.0;
                deal_public_card(
                    shoe,
                    network,
                    &mut hands[index],
                    dealer_upcard,
                    Phase::Observe,
                )?;
                return Ok(());
            }
            Action::Split => {
                let mut split_hand = hands[index].split_off_pair();
                deal_public_card(
                    shoe,
                    network,
                    &mut hands[index],
                    dealer_upcard,
                    Phase::Observe,
                )?;
                deal_public_card(
                    shoe,
                    network,
                    &mut split_hand,
                    dealer_upcard,
                    Phase::Observe,
                )?;
                hands.push(split_hand);
                if hands[index].cards[0] == 1 {
                    return Ok(());
                }
            }
            Action::Surrender => {
                hands[index].surrendered = true;
                return Ok(());
            }
            Action::None => return Ok(()),
        }
    }
}

fn legal_actions(hand: &Hand, hand_count: usize) -> Vec<Action> {
    let mut actions = Vec::from([Action::Stand]);
    if hand.cards.len() < MAX_PLAYER_CARDS {
        actions.push(Action::Hit);
    }
    if hand.cards.len() == 2 && !hand.split {
        actions.push(Action::Surrender);
    }
    if hand.cards.len() == 2 {
        actions.push(Action::Double);
    }
    if hand.cards.len() == 2 && hand.pair_rank() > 0 && hand_count <= MAX_SPLITS {
        actions.push(Action::Split);
    }
    actions
}

fn settle_all(player_hands: &[Hand], dealer: &Hand, report: &mut ShoeReport) {
    report.hands += player_hands.len();
    for player in player_hands {
        if player.doubled {
            report.doubled += 1;
        }
        if player.split {
            report.split += 1;
        }
        if player.surrendered {
            report.surrendered += 1;
            report.losses += 1;
            report.profit -= player.bet * 0.5;
            continue;
        }
        if player.is_bust() {
            report.busts += 1;
            report.losses += 1;
            report.profit -= player.bet;
            continue;
        }
        if player.is_blackjack() && !dealer.is_blackjack() {
            report.blackjacks += 1;
            report.wins += 1;
            report.profit += player.bet * 1.5;
            continue;
        }
        if dealer.is_bust() || player.value() > dealer.value() {
            report.wins += 1;
            report.profit += player.bet;
        } else if player.value() < dealer.value() || dealer.is_blackjack() {
            report.losses += 1;
            report.profit -= player.bet;
        } else {
            report.pushes += 1;
        }
    }
}

impl ShoeReport {
    fn absorb(&mut self, other: Self) {
        self.profit += other.profit;
        self.wins += other.wins;
        self.losses += other.losses;
        self.pushes += other.pushes;
        self.busts += other.busts;
        self.blackjacks += other.blackjacks;
        self.doubled += other.doubled;
        self.split += other.split;
        self.surrendered += other.surrendered;
        self.rounds += other.rounds;
        self.hands += other.hands;
    }
}

fn deal_public_card(
    shoe: &mut Shoe,
    network: &mut dyn BlackjackPolicy,
    hand: &mut Hand,
    dealer_upcard: u8,
    phase: Phase,
) -> Result<(), FitnessError> {
    let card = draw_or_err(shoe)?;
    hand.add(card);
    observe_card(
        network,
        hand,
        dealer_upcard,
        Action::None,
        phase,
        card,
        shoe,
    )
}

fn observe_hand(
    network: &mut dyn BlackjackPolicy,
    hand: &Hand,
    dealer_upcard: u8,
    last_action: Action,
    phase: Phase,
    shoe: &Shoe,
) -> Result<(), FitnessError> {
    for card in &hand.cards {
        let state = DecisionState {
            total: hand.value(),
            dealer_upcard,
            soft: hand.is_soft(),
            pair_rank: hand.pair_rank(),
            card_count: hand.cards.len() as u8,
            can_hit: false,
            can_double: false,
            can_split: false,
            can_surrender: false,
            last_action,
            phase,
            exposed_card: *card,
            shoe_progress: shoe.progress(),
        };
        network.activate(&state.inputs())?;
    }
    Ok(())
}

fn observe_card(
    network: &mut dyn BlackjackPolicy,
    hand: &Hand,
    dealer_upcard: u8,
    last_action: Action,
    phase: Phase,
    exposed_card: u8,
    shoe: &Shoe,
) -> Result<(), FitnessError> {
    let state = DecisionState {
        total: hand.value(),
        dealer_upcard,
        soft: hand.is_soft(),
        pair_rank: hand.pair_rank(),
        card_count: hand.cards.len() as u8,
        can_hit: false,
        can_double: false,
        can_split: false,
        can_surrender: false,
        last_action,
        phase,
        exposed_card,
        shoe_progress: shoe.progress(),
    };
    network.activate(&state.inputs())?;
    Ok(())
}

fn state_for(
    hand: &Hand,
    hand_count: usize,
    dealer_upcard: u8,
    last_action: Action,
    phase: Phase,
    exposed_card: u8,
    shoe: &Shoe,
) -> DecisionState {
    let legal = legal_actions(hand, hand_count);
    DecisionState {
        total: hand.value(),
        dealer_upcard,
        soft: hand.is_soft(),
        pair_rank: hand.pair_rank(),
        card_count: hand.cards.len() as u8,
        can_hit: legal.contains(&Action::Hit),
        can_double: legal.contains(&Action::Double),
        can_split: legal.contains(&Action::Split),
        can_surrender: legal.contains(&Action::Surrender),
        last_action,
        phase,
        exposed_card,
        shoe_progress: shoe.progress(),
    }
}

fn dealer_should_hit(dealer: &Hand) -> bool {
    dealer.value() < 17
}

fn dealer_may_peek(upcard: u8) -> bool {
    upcard == 1 || upcard == 10
}

fn draw_or_err(shoe: &mut Shoe) -> Result<u8, FitnessError> {
    shoe.draw()
        .ok_or_else(|| FitnessError::new("blackjack shoe ran out of cards"))
}

fn card_score(card: u8) -> u8 {
    if card == 1 {
        11
    } else {
        card
    }
}

fn bool_input(value: bool) -> f64 {
    if value {
        1.0
    } else {
        -1.0
    }
}

fn evaluate_report(
    genome: &DefaultGenome,
    config: &Config,
    target_rounds: usize,
) -> Result<ShoeReport, FitnessError> {
    let mut total = ShoeReport::default();
    let mut shoes = 0;
    while total.rounds < target_rounds.max(1) {
        let seed = REPORT_SEEDS[shoes % REPORT_SEEDS.len()]
            + ((shoes / REPORT_SEEDS.len()) as u64 * 10_007);
        let mut network = RecurrentNetwork::create(genome, &config.genome)
            .map_err(|err| FitnessError::new(err.to_string()))?;
        let report = run_shoe(GenomeId::new(0), seed, &mut network)?;
        total.fitness += report.fitness;
        total.profit += report.profit;
        total.wins += report.wins;
        total.losses += report.losses;
        total.pushes += report.pushes;
        total.busts += report.busts;
        total.blackjacks += report.blackjacks;
        total.doubled += report.doubled;
        total.split += report.split;
        total.surrendered += report.surrendered;
        total.rounds += report.rounds;
        total.hands += report.hands;
        shoes += 1;
    }

    Ok(ShoeReport {
        fitness: total.fitness / shoes as f64,
        profit: total.profit / shoes as f64,
        ..total
    })
}

fn evaluate_linear_params(params: &[f64], target_rounds: usize, seed_base: u64) -> ShoeReport {
    let mut total = ShoeReport::default();
    let mut shoes = 0usize;
    while total.rounds < target_rounds.max(1) {
        let seed = TRAIN_SHOES[shoes % TRAIN_SHOES.len()]
            + seed_base
            + ((shoes / TRAIN_SHOES.len()) as u64 * 10_007);
        let mut policy = LinearBlackjackPolicy::new(params);
        match run_shoe(GenomeId::new(0), seed, &mut policy) {
            Ok(report) => total.absorb(report),
            Err(_) => {
                total.losses += 1;
                total.rounds += 1;
                total.profit -= 1.0;
            }
        }
        shoes += 1;
    }
    if shoes > 0 {
        total.fitness /= shoes as f64;
        total.profit /= shoes as f64;
    }
    total
}

fn linear_train_fitness(params: &[f64], seed_base: u64) -> f64 {
    let mut total = 0.0;
    for seed in TRAIN_SHOES {
        let mut policy = LinearBlackjackPolicy::new(params);
        match run_shoe(GenomeId::new(0), seed ^ seed_base, &mut policy) {
            Ok(report) => total += report.fitness,
            Err(_) => return 0.0,
        }
    }
    total / TRAIN_SHOES.len() as f64
}

#[derive(Clone)]
struct LinearBlackjackOpenEsEval {
    base_seed: u64,
}

impl evo_optim_rust::OpenEsEvaluator for LinearBlackjackOpenEsEval {
    fn eval_test(&self, parameters: &[f64]) -> f64 {
        linear_train_fitness(parameters, self.base_seed)
    }

    fn eval_train(&self, parameters: &[f64], loop_index: usize) -> f64 {
        linear_train_fitness(parameters, self.base_seed + loop_index as u64 * 997)
    }
}

fn initial_linear_params(base_seed: u64) -> Vec<f64> {
    let mut initial_rng = neat_rust::algorithm::XorShiftRng::seed_from_u64(base_seed);
    let mut initial_params = vec![0.0; LINEAR_PARAM_COUNT];
    for param in &mut initial_params {
        *param = (initial_rng.next_f64() * 2.0 - 1.0) * 0.05;
    }
    initial_params
}

fn write_linear_summary(
    output_dir: &Path,
    optimizer: &str,
    base_seed: u64,
    generations: usize,
    population_size: usize,
    train_rounds: usize,
    best_score: f64,
    best_params: &[f64],
) -> Result<(), Box<dyn Error>> {
    let report = evaluate_linear_params(best_params, DEFAULT_REPORT_ROUNDS, base_seed + 99_000);
    let resolved = (report.wins + report.losses + report.pushes).max(1);
    let ev_per_round = report.profit / report.rounds.max(1) as f64;
    fs::write(
        output_dir.join("models").join("winner_params.json"),
        format!("{}\n", serde_json::to_string_pretty(best_params)?),
    )?;
    let summary = serde_json::json!({
        "sample": "blackjack",
        "optimizer": optimizer,
        "base_seed": base_seed,
        "generations": generations,
        "population_size": population_size,
        "report_rounds_target": train_rounds,
        "params": LINEAR_PARAM_COUNT,
        "best_score": best_score,
        "report": {
            "fitness": report.fitness,
            "avg_profit_per_shoe": report.profit,
            "ev_per_round": ev_per_round,
            "win_rate": report.wins as f64 / resolved as f64,
            "rounds": report.rounds,
            "hands": report.hands,
            "wins": report.wins,
            "losses": report.losses,
            "pushes": report.pushes,
            "busts": report.busts,
            "blackjacks": report.blackjacks,
            "doubled": report.doubled,
            "split": report.split,
            "surrendered": report.surrendered
        }
    });
    fs::write(
        output_dir.join("run_summary.json"),
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )?;
    println!(
        "blackjack optimizer={} generations={} population={} report_rounds={} best_score={:.2} ev_per_round={:.4} win_rate={:.3} artifacts_dir={}",
        optimizer,
        generations,
        population_size,
        train_rounds,
        best_score,
        ev_per_round,
        report.wins as f64 / resolved as f64,
        output_dir.display()
    );
    Ok(())
}

fn run_openes_mode(args: &[String]) -> Result<(), Box<dyn Error>> {
    let generations = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let train_rounds = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPORT_ROUNDS);
    let population_size = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(72);
    let base_seed = args
        .get(4)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_BASE_SEED);
    let output_dir = args.get(5).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("..")
            .join("logs")
            .join("ES")
            .join(format!("blackjack_openes_seed{base_seed}"))
    });
    prepare_output_dir(&output_dir)?;

    let samples = (population_size / 2).max(1);
    let mut adam = evo_optim_rust::Adam::default();
    adam.set_lr(0.03);
    let mut openes = evo_optim_rust::OpenEs::new(adam, LinearBlackjackOpenEsEval { base_seed });
    openes
        .set_params(initial_linear_params(base_seed))
        .set_std(0.05)
        .set_samples(samples);

    let checkpointer = evo_optim_rust::Checkpointer::new(
        Some(1),
        output_dir
            .join("checkpoints")
            .join("openes_gen")
            .to_string_lossy(),
    );
    let metrics_path = output_dir.join("metrics").join("openes.ndjson");
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = openes.get_params().to_vec();
    for generation in 0..generations {
        let candidates = openes.ask(base_seed + generation as u64);
        let scores = candidates
            .iter()
            .map(|candidate| {
                let score =
                    linear_train_fitness(&candidate.params, base_seed + generation as u64 * 997);
                if score > best_score {
                    best_score = score;
                    best_params = candidate.params.clone();
                }
                evo_optim_rust::CandidateScore {
                    sample_index: candidate.sample_index,
                    sign: candidate.sign,
                    score,
                }
            })
            .collect::<Vec<_>>();
        let report = openes.tell(
            &candidates,
            &scores,
            evo_optim_rust::ScoreTransform::CenteredRank,
        );
        let metrics = evo_optim_rust::OpenEsGenerationMetrics::from_report(
            &report,
            &candidates,
            &scores,
            openes.get_params(),
        );
        evo_optim_rust::append_jsonl(&metrics_path, &metrics)?;
        checkpointer.save_checkpoint(
            openes.generation(),
            &evo_optim_rust::OpenEsCheckpointState {
                generation: openes.generation(),
                params: openes.get_params().to_vec(),
                best_score,
                best_params: best_params.clone(),
                samples,
                noise_std: openes.noise_std(),
            },
        )?;
        println!(
            "[generation blackjack-openes] gen={} best={:.3} mean={:.3} grad_norm={:.4}",
            metrics.generation, best_score, metrics.mean_score, metrics.gradient_norm
        );
    }
    write_linear_summary(
        &output_dir,
        "openes",
        base_seed,
        generations,
        samples * 2,
        train_rounds,
        best_score,
        &best_params,
    )
}

fn run_cma_mode(args: &[String]) -> Result<(), Box<dyn Error>> {
    let generations = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let train_rounds = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPORT_ROUNDS);
    let population_size = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(72);
    let base_seed = args
        .get(4)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_BASE_SEED);
    let output_dir = args.get(5).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("..")
            .join("logs")
            .join("ES")
            .join(format!("blackjack_cma_seed{base_seed}"))
    });
    prepare_output_dir(&output_dir)?;

    let param_space = evo_optim_rust::ParamSpace {
        dimension: LINEAR_PARAM_COUNT,
        init_kind: evo_optim_rust::ParamInitKind::Uniform,
        init_scale: 0.05,
        min_value: Some(-4.0),
        max_value: Some(4.0),
    };
    let initial_params = initial_linear_params(base_seed);

    let mut options = evo_optim_rust::CmaRunnerOptions::generations(1);
    options.seed = base_seed;
    options.population_size = Some(population_size);
    options.initial_sigma = 0.5;
    options.checkpointer = Some(evo_optim_rust::Checkpointer::new(
        Some(1),
        output_dir
            .join("checkpoints")
            .join("cma_gen")
            .to_string_lossy(),
    ));
    options.metrics_path = Some(output_dir.join("metrics").join("cma.ndjson"));

    let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let eval_counter_closure = std::sync::Arc::clone(&eval_counter);
    let evaluator = move |params: &[f64]| {
        let eval_id = eval_counter_closure.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        linear_train_fitness(params, base_seed + eval_id * 997)
    };
    let mut runner =
        evo_optim_rust::CmaTrainRunner::new(param_space, initial_params, evaluator, options)?;
    for _ in 0..generations {
        let termination = runner.run_generation()?;
        let metrics = runner.metrics();
        println!(
            "[generation blackjack-cma] gen={} best={:.3} current_best={:.3} sigma={:.4}",
            metrics.generation, metrics.best_score, metrics.current_best_score, metrics.sigma
        );
        if termination.is_some() {
            break;
        }
    }
    let checkpoint = runner.checkpoint();
    write_linear_summary(
        &output_dir,
        "cma",
        base_seed,
        generations,
        population_size,
        train_rounds,
        checkpoint.best_score,
        &checkpoint.best_params,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args
        .first()
        .is_some_and(|value| value.eq_ignore_ascii_case("cma"))
    {
        return run_cma_mode(&args);
    }
    if args
        .first()
        .is_some_and(|value| value.eq_ignore_ascii_case("openes"))
    {
        return run_openes_mode(&args);
    }
    let generations = args
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12);
    let bootstrap_rounds = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let profile = BlackjackConfigProfile::parse(args.get(2).map(String::as_str))?;
    let report_rounds = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPORT_ROUNDS);
    let output_or_seed = args.get(4);
    let base_seed = args
        .get(5)
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| output_or_seed.and_then(|value| value.parse::<u64>().ok()))
        .unwrap_or(DEFAULT_BASE_SEED);
    let output_dir = output_or_seed
        .filter(|value| value.parse::<u64>().is_err())
        .map(PathBuf::from)
        .unwrap_or_else(|| default_output_dir(profile, base_seed));
    let config_path = profile.config_path();
    let config = Config::from_file(&config_path)?;
    let mut evaluator = BlackjackEvaluator;
    prepare_output_dir(&output_dir)?;
    let mut best = run_population(
        Population::new(config.clone(), base_seed)?,
        &mut evaluator,
        generations,
        &output_dir,
        &config_path,
        "round0",
    )?;
    print_report(
        profile,
        "round=0 bootstrap=false",
        generations,
        &best,
        &config,
        report_rounds,
    )?;

    for round in 1..=bootstrap_rounds {
        let strategy = BootstrapStrategy::from_champion(best.clone(), 0.5);
        let candidate = run_population(
            Population::new_with_bootstrap(config.clone(), base_seed + round as u64, strategy)?,
            &mut evaluator,
            generations,
            &output_dir,
            &config_path,
            &format!("round{round}"),
        )?;
        print_report(
            profile,
            &format!("round={round} bootstrap=true"),
            generations,
            &candidate,
            &config,
            report_rounds,
        )?;
        if candidate.fitness.unwrap_or(f64::NEG_INFINITY)
            > best.fitness.unwrap_or(f64::NEG_INFINITY)
        {
            best = candidate;
        }
    }

    let final_report = print_report(profile, "final", generations, &best, &config, report_rounds)?;
    write_final_outputs(
        &output_dir,
        profile,
        generations,
        bootstrap_rounds,
        report_rounds,
        base_seed,
        &best,
        &config,
        &final_report,
    )?;
    println!("blackjack artifacts_dir={}", output_dir.display());
    Ok(())
}

fn run_population(
    mut population: Population,
    evaluator: &mut BlackjackEvaluator,
    generations: usize,
    output_dir: &Path,
    config_path: &Path,
    run_label: &str,
) -> Result<DefaultGenome, Box<dyn Error>> {
    let checkpoint_prefix = output_dir
        .join("checkpoints")
        .join(format!("{run_label}-neat-rust-checkpoint-gen"));
    population.checkpoint_sink = Some(new_rust_checkpoint_sink(
        Some(5),
        checkpoint_prefix.to_string_lossy().to_string(),
        config_path.to_path_buf(),
    ));
    Ok(population
        .run_with_evaluator(evaluator, Some(generations))?
        .expect("population should keep a champion"))
}

fn print_report(
    profile: BlackjackConfigProfile,
    label: &str,
    generations: usize,
    genome: &DefaultGenome,
    config: &Config,
    report_rounds: usize,
) -> Result<ShoeReport, Box<dyn Error>> {
    let report = evaluate_report(genome, config, report_rounds)?;
    let resolved = (report.wins + report.losses + report.pushes).max(1);
    let win_rate = report.wins as f64 / resolved as f64;
    let ev_per_round = report.profit / report.rounds.max(1) as f64;
    println!(
        "blackjack profile={} {label} generations={generations} best_genome={} train_fitness={:.2} report_fitness={:.2} avg_profit={:.2} ev_per_round={:.4} win_rate={:.3} target_rounds={} rounds={} hands={} busts={} blackjacks={} doubled={} split={} surrendered={}",
        profile.name(),
        genome.key,
        genome.fitness.unwrap_or(0.0),
        report.fitness,
        report.profit,
        ev_per_round,
        win_rate,
        report_rounds,
        report.rounds,
        report.hands,
        report.busts,
        report.blackjacks,
        report.doubled,
        report.split,
        report.surrendered,
    );
    Ok(report)
}

fn default_output_dir(profile: BlackjackConfigProfile, base_seed: u64) -> PathBuf {
    PathBuf::from("..")
        .join("logs")
        .join("NEAT")
        .join(format!("blackjack_{}_seed{base_seed}", profile.name()))
}

fn prepare_output_dir(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir.join("checkpoints"))?;
    fs::create_dir_all(output_dir.join("models"))?;
    Ok(())
}

fn write_final_outputs(
    output_dir: &Path,
    profile: BlackjackConfigProfile,
    generations: usize,
    bootstrap_rounds: usize,
    report_rounds: usize,
    base_seed: u64,
    genome: &DefaultGenome,
    config: &Config,
    report: &ShoeReport,
) -> Result<(), Box<dyn Error>> {
    let model_json = export_neat_genome_json(genome, config, "blackjack14");
    fs::write(
        output_dir.join("models").join("winner_genome.json"),
        model_json,
    )?;

    let resolved = (report.wins + report.losses + report.pushes).max(1);
    let ev_per_round = report.profit / report.rounds.max(1) as f64;
    let summary = serde_json::json!({
        "sample": "blackjack",
        "profile": profile.name(),
        "base_seed": base_seed,
        "generations": generations,
        "bootstrap_rounds": bootstrap_rounds,
        "checkpoint_interval": 5,
        "report_rounds_target": report_rounds,
        "best_genome": genome.key.raw(),
        "train_fitness": genome.fitness.unwrap_or(0.0),
        "report": {
            "fitness": report.fitness,
            "avg_profit_per_shoe": report.profit,
            "ev_per_round": ev_per_round,
            "win_rate": report.wins as f64 / resolved as f64,
            "rounds": report.rounds,
            "hands": report.hands,
            "wins": report.wins,
            "losses": report.losses,
            "pushes": report.pushes,
            "busts": report.busts,
            "blackjacks": report.blackjacks,
            "doubled": report.doubled,
            "split": report.split,
            "surrendered": report.surrendered
        }
    });
    fs::write(
        output_dir.join("run_summary.json"),
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )?;
    Ok(())
}
