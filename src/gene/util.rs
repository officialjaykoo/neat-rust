use crate::attributes::RandomSource;

pub(super) fn choose_copy<T: Copy>(first: T, second: T, rng: &mut impl RandomSource) -> T {
    if rng.next_f64() > 0.5 {
        first
    } else {
        second
    }
}
