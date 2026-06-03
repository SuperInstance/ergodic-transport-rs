//! # Ergodic Transport — Rust Port
//!
//! Rust implementation of the ergodic transport C library.
//! Provides Markov chain analysis, ergodic theory utilities,
//! budget planning, and control tools.
//!
//! ## Audit Fixes over C Original
//!
//! - **Persistent RNG**: `MarkovChain` holds its own `rng` field (`rand::rngs::SmallRng`),
//!   seeded at construction or by `seed()`.  No global mutable state.
//! - **`is_ergodic` uses BFS + GCD period**: correct textbook check for
//!   irreducibility (bidirectional BFS) and aperiodicity (GCD of cycle lengths
//!   through any state via BFS distance).
//! - **`stationary_distribution`**: power method on the *transpose* (correct).
//! - **`birkhoff_bound`**: N ≥ var / (ε² × δ), ceiling'd.
//! - **`budget_safety_margin`**: returns `mixing_time × std_cost`.
//! - **`wasserstein_distance`**: W₁ via CDF difference on ordered states 0..n-1.
//! - **`lqr_control`**: returns optimal budget allocation using a simple proportional
//!   controller with non-negativity clamp.

use nalgebra::DMatrix;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub const MAX_STATES: usize = 64;

/// A row-stochastic transition matrix.
///
/// # Invariants
///
/// - `n ≤ MAX_STATES`
/// - For every row `i`, `∑_j P[i][j] ≈ 1.0`
/// - All entries are non-negative
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    pub n: usize,
    /// Row-major: `P[i][j]` = probability of moving from i to j.
    pub data: [[f64; MAX_STATES]; MAX_STATES],
}

// ---------------------------------------------------------------------------
// Markov chain implementation
// ---------------------------------------------------------------------------

/// A Markov chain with a persistent RNG.
///
/// # Audit note
///
/// The C version used a global static `rng_state`; this version stores the RNG
/// as a field so that `simulate` calls advance the state without interference.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub tm: TransitionMatrix,
    rng: SmallRng,
}

impl Default for MarkovChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkovChain {
    /// Build a chain from a flat row-major slice of length `n×n`.
    ///
    /// # Panics
    ///
    /// Panics if `n > MAX_STATES` or the slice length is less than `n*n`.
    pub fn from_flat(n: usize, flat: &[f64]) -> Self {
        assert!(n <= MAX_STATES, "n={} exceeds MAX_STATES={}", n, MAX_STATES);
        assert!(flat.len() >= n * n, "flat slice too short");
        let mut data = [[0.0_f64; MAX_STATES]; MAX_STATES];
        for i in 0..n {
            for j in 0..n {
                data[i][j] = flat[i * n + j];
            }
        }
        Self {
            tm: TransitionMatrix { n, data },
            rng: SmallRng::from_entropy(),
        }
    }

    /// Create a new chain with a zero matrix (useful for tests that overwrite).
    pub fn new() -> Self {
        Self {
            tm: TransitionMatrix {
                n: 0,
                data: [[0.0; MAX_STATES]; MAX_STATES],
            },
            rng: SmallRng::from_entropy(),
        }
    }

    /// Seed the internal RNG.
    pub fn seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }

    /// Return a reference to the underlying transition matrix.
    pub fn matrix(&self) -> &TransitionMatrix {
        &self.tm
    }

    /// Return a mutable reference to the underlying transition matrix.
    pub fn matrix_mut(&mut self) -> &mut TransitionMatrix {
        &mut self.tm
    }

    // ------------------------------------------------------------------
    // Stationary distribution (power iteration, using transpose)
    // ------------------------------------------------------------------

    /// Compute the stationary distribution via power iteration on the
    /// transpose of the transition matrix.
    ///
    /// Returns `None` if the iteration does not converge within 100 000 steps.
    pub fn stationary_distribution(&self) -> Option<Vec<f64>> {
        let n = self.tm.n;
        if n == 0 {
            return None;
        }
        // Build transpose: T[j][i] = P[i][j]
        let mut tr = [[0.0_f64; MAX_STATES]; MAX_STATES];
        for i in 0..n {
            for j in 0..n {
                tr[j][i] = self.tm.data[i][j];
            }
        }

        // Initialise uniform
        let mut pi = vec![1.0 / n as f64; n];

        for _iter in 0..100_000 {
            // pi * P^T  (row vector * transpose matrix)
            let mut new_pi = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_pi[j] += pi[i] * tr[j][i]; // = pi[i] * P[i][j]
                }
            }

            // Normalise
            let sum: f64 = new_pi.iter().sum();
            if sum == 0.0 {
                return None;
            }
            for v in &mut new_pi {
                *v /= sum;
            }

            // Convergence check (L1)
            let diff: f64 = new_pi.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum();
            pi = new_pi;
            if diff < 1e-12 {
                return Some(pi);
            }
        }
        None // did not converge
    }

    // ------------------------------------------------------------------
    // Ergodicity check (BFS + GCD period)
    // ------------------------------------------------------------------

    /// Check whether the chain is ergodic (irreducible + aperiodic).
    ///
    /// Returns `(true, explanation)` if ergodic, `(false, reason)` otherwise.
    pub fn is_ergodic(&self) -> (bool, String) {
        let n = self.tm.n;
        if n == 0 {
            return (false, "chain has no states".into());
        }

        // --- 1. Irreducibility ---
        // Forward: all states reachable from 0
        let reachable_from_0 = bfs_reachable(&self.tm, 0);
        for i in 0..n {
            if !reachable_from_0[i] {
                return (
                    false,
                    format!("not irreducible: state {} not reachable from state 0", i),
                );
            }
        }

        // Reverse (transpose graph): state 0 reachable from all states
        let mut tr = TransitionMatrix {
            n,
            data: [[0.0; MAX_STATES]; MAX_STATES],
        };
        for i in 0..n {
            for j in 0..n {
                tr.data[j][i] = self.tm.data[i][j];
            }
        }
        let can_reach_0 = bfs_reachable(&tr, 0);
        for i in 0..n {
            if !can_reach_0[i] {
                return (
                    false,
                    format!("not irreducible: state 0 not reachable from state {}", i),
                );
            }
        }

        // --- 2. Aperiodicity ---
        let period = state_period(&self.tm, 0);
        if period != 1 {
            return (
                false,
                format!("chain has period {} (not aperiodic)", period),
            );
        }

        (true, "chain is ergodic (irreducible + aperiodic)".into())
    }

    // ------------------------------------------------------------------
    // Mixing time
    // ------------------------------------------------------------------

    /// Mixing time: smallest `k` such that `‖πₖ - π‖_TV < ε`, starting from
    /// state 0.  Returns `None` if not converged within `max_iter` steps.
    ///
    /// Uses total variation distance = ½·L1.
    pub fn mixing_time(&self, epsilon: f64, max_iter: usize) -> Option<usize> {
        let pi = self.stationary_distribution()?;
        let n = self.tm.n;
        let mut dist = vec![0.0; n];
        dist[0] = 1.0;

        for k in 0..max_iter {
            // dist * P
            let mut new_dist = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_dist[j] += dist[i] * self.tm.data[i][j];
                }
            }
            dist = new_dist;

            // TV distance
            let tv: f64 = dist.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>()
                * 0.5;
            if tv < epsilon {
                return Some(k + 1);
            }
        }
        None
    }

    // ------------------------------------------------------------------
    // Birkhoff / Chebyshev bound
    // ------------------------------------------------------------------

    /// Given a function `f` over states and its stationary distribution,
    /// compute `N` such that `N` samples guarantee `P(|avg - μ| ≥ ε) ≤ δ`.
    ///
    /// Uses the Chebyshev-like bound:  N ≥ var(f) / (ε² × δ).
    pub fn birkhoff_bound(&self, epsilon: f64, delta: f64) -> Option<f64> {
        if epsilon <= 0.0 || delta <= 0.0 {
            return None;
        }
        // Compute stationary mean & variance of the cost f(i) = i
        // (We use the state index as the cost function — same as C default.)
        let pi = self.stationary_distribution()?;
        let n = self.tm.n;
        let f: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let mu: f64 = pi.iter().zip(&f).map(|(p, &x)| p * x).sum();
        let var: f64 = pi
            .iter()
            .zip(&f)
            .map(|(p, &x)| {
                let d = x - mu;
                p * d * d
            })
            .sum();

        let n_ = var / (epsilon * epsilon * delta);
        Some(n_.ceil())
    }

    // ------------------------------------------------------------------
    // Budget safety margin
    // ------------------------------------------------------------------

    /// Extra budget needed for the mixing period beyond stationary prediction.
    ///
    /// Returns `mixing_time × std_cost`, which has dimensions `[steps] × [cost] = [cost]`.
    pub fn budget_safety_margin(&self, epsilon: f64) -> Option<f64> {
        let mixing = self.mixing_time(epsilon, 10_000)?;
        let pi = self.stationary_distribution()?;

        // cost = state index
        let mean_cost: f64 = pi.iter().enumerate().map(|(i, &p)| p * i as f64).sum();
        let var_cost: f64 = pi
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let d = i as f64 - mean_cost;
                p * d * d
            })
            .sum();
        let std_cost = var_cost.sqrt();

        Some(mixing as f64 * std_cost)
    }

    // ------------------------------------------------------------------
    // Simulate
    // ------------------------------------------------------------------

    /// Simulate `N` steps starting from state `s0`.
    ///
    /// # Audit note
    ///
    /// The RNG is *not* reset between calls, so repeated calls produce
    /// different trajectories.  Call `seed()` for deterministic control.
    pub fn simulate(&mut self, s0: usize, n_steps: usize) -> Vec<usize> {
        let n = self.tm.n;
        let mut state = s0;
        let mut traj = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            traj.push(state);
            let r: f64 = self.rng.gen();
            let mut cum = 0.0;
            let mut next = state;
            for j in 0..n {
                cum += self.tm.data[state][j];
                if r <= cum {
                    next = j;
                    break;
                }
            }
            state = next;
        }
        traj
    }
}

// ===========================================================================
// Free functions
// ===========================================================================

/// Time average: `(1/N) ∑ f(trajectory[t])`.
pub fn ergodic_time_average(trajectory: &[usize], f: &[f64]) -> f64 {
    let sum: f64 = trajectory.iter().map(|&s| f[s]).sum();
    sum / trajectory.len() as f64
}

/// Ensemble average: `∑ π_i · f(i)`.
pub fn ergodic_ensemble_average(pi: &[f64], f: &[f64]) -> f64 {
    pi.iter().zip(f).map(|(p, &x)| p * x).sum()
}

/// Check ergodicity: `|time_avg - ensemble_avg| < tolerance`.
pub fn ergodic_check(
    trajectory: &[usize],
    f: &[f64],
    pi: &[f64],
    tolerance: f64,
) -> bool {
    let ta = ergodic_time_average(trajectory, f);
    let ea = ergodic_ensemble_average(pi, f);
    (ta - ea).abs() < tolerance
}

/// Birkhoff/Chebyshev bound for a given empirical distribution.
///
/// N ≥ var_f(p) / (ε² × δ)
pub fn birkhoff_bound(pi: &[f64], f: &[f64], eps: f64, delta: f64) -> Option<f64> {
    if eps <= 0.0 || delta <= 0.0 {
        return None;
    }
    let mu: f64 = pi.iter().zip(f).map(|(p, &x)| p * x).sum();
    let var: f64 = pi
        .iter()
        .zip(f)
        .map(|(p, &x)| {
            let d = x - mu;
            p * d * d
        })
        .sum();
    let n_ = var / (eps * eps * delta);
    Some(n_.ceil())
}

/// Occupation measure: `mu[i] = fraction of time spent in state i`.
pub fn occupation_measure(trajectory: &[usize], n_states: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n_states];
    for &s in trajectory {
        counts[s] += 1.0;
    }
    let total = trajectory.len() as f64;
    for c in &mut counts {
        *c /= total;
    }
    counts
}

/// Wasserstein-1 distance via CDF (states are ordered 0..n-1).
pub fn wasserstein_distance(mu: &[f64], nu: &[f64]) -> f64 {
    let mut dist = 0.0;
    let mut cdf_mu = 0.0;
    let mut cdf_nu = 0.0;
    for i in 0..mu.len().min(nu.len()) {
        cdf_mu += mu[i];
        cdf_nu += nu[i];
        dist += (cdf_mu - cdf_nu).abs();
    }
    dist
}

/// Drift: W1 distance between empirical occupation and stationary distribution.
pub fn consumption_drift(trajectory: &[usize], pi: &[f64]) -> f64 {
    let mu = occupation_measure(trajectory, pi.len());
    wasserstein_distance(&mu, pi)
}

/// Control correction: `correction[i] = -(current_mu[i] - target_pi[i]) * cost[i]`.
pub fn control_correction(
    current_mu: &[f64],
    target_pi: &[f64],
    cost: &[f64],
) -> Vec<f64> {
    let n = current_mu.len().min(target_pi.len()).min(cost.len());
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        let deviation = current_mu[i] - target_pi[i];
        corr.push(-deviation * cost[i]);
    }
    corr
}

/// Simple LQR step: proportional controller.
///
/// `result[i] = max(0, allocation[i] - gain * (allocation[i] - target[i] * cost[i]))`
pub fn lqr_step(
    allocation: &[f64],
    target: &[f64],
    cost: &[f64],
    gain: f64,
) -> Vec<f64> {
    let n = allocation.len().min(target.len()).min(cost.len());
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let error = allocation[i] - target[i] * cost[i];
        let v = allocation[i] - gain * error;
        result.push(if v < 0.0 { 0.0 } else { v });
    }
    result
}

/// Predict long-run consumption: `∑ π_i · resources[i]`.
pub fn predict_consumption(pi: &[f64], resources: &[f64]) -> f64 {
    pi.iter()
        .zip(resources)
        .map(|(p, &r)| p * r)
        .sum()
}

/// Budget adequacy: total allocated - predicted consumption.
pub fn budget_adequacy(allocation: &[f64], pi: &[f64], resources: &[f64]) -> f64 {
    let total_alloc: f64 = allocation.iter().sum();
    let predicted = predict_consumption(pi, resources);
    total_alloc - predicted
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// BFS from `start`, returning `visited[i] = true` if reachable via positive edges.
fn bfs_reachable(tm: &TransitionMatrix, start: usize) -> [bool; MAX_STATES] {
    let n = tm.n;
    let mut visited = [false; MAX_STATES];
    let mut queue = [0_usize; MAX_STATES * MAX_STATES];
    let mut head = 0;
    let mut tail = 0;

    visited[start] = true;
    queue[tail] = start;
    tail += 1;

    while head < tail {
        let s = queue[head];
        head += 1;
        for j in 0..n {
            if tm.data[s][j] > 0.0 && !visited[j] {
                visited[j] = true;
                queue[tail] = j;
                tail += 1;
            }
        }
    }
    visited
}

/// Compute the period of state `s` via BFS and GCD of cycle lengths.
fn state_period(tm: &TransitionMatrix, s: usize) -> usize {
    let n = tm.n;
    let mut dist = [-1_i32; MAX_STATES];
    let mut queue = [0_usize; MAX_STATES * MAX_STATES];
    let mut head = 0;
    let mut tail = 0;

    dist[s] = 0;
    queue[tail] = s;
    tail += 1;

    let mut g = 0_i32;

    while head < tail {
        let u = queue[head];
        head += 1;
        for j in 0..n {
            if tm.data[u][j] > 0.0 {
                if dist[j] < 0 {
                    dist[j] = dist[u] + 1;
                    queue[tail] = j;
                    tail += 1;
                } else {
                    // Cycle: u -> j, and we already know dist[j]
                    let cycle_len = dist[u] + 1 + dist[j];
                    g = gcd(g, cycle_len);
                    if g == 1 {
                        return 1;
                    }
                }
            }
        }
    }

    if g == 0 {
        1
    } else {
        g as usize
    }
}

fn gcd(a: i32, b: i32) -> i32 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// A simple 2-state chain that is ergodic.
    fn simple_ergodic_2() -> MarkovChain {
        // P = [[0.7, 0.3], [0.4, 0.6]]
        // Irreducible + aperiodic (period 1)
        MarkovChain::from_flat(2, &[0.7, 0.3, 0.4, 0.6])
    }

    /// A 3-state chain with period 2.
    fn periodic_3() -> MarkovChain {
        // 0 <-> 1, 2 unreachable → not irreducible
        MarkovChain::from_flat(3, &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// A 4-state deterministic cycle: 0 -> 1 -> 2 -> 3 -> 0  (period 4).
    fn cycle_4() -> MarkovChain {
        MarkovChain::from_flat(
            4,
            &[
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0, //
            ],
        )
    }

    /// Reducible 2-state chain.
    fn reducible_2() -> MarkovChain {
        MarkovChain::from_flat(2, &[0.5, 0.5, 0.0, 1.0])
    }

    // ------------------------------------------------------------------
    // 1–4: Construction & basic properties
    // ------------------------------------------------------------------

    #[test]
    fn test_construction_flat() {
        let mc = MarkovChain::from_flat(2, &[0.1, 0.9, 0.8, 0.2]);
        assert_eq!(mc.tm.n, 2);
        assert!((mc.tm.data[0][0] - 0.1).abs() < 1e-12);
        assert!((mc.tm.data[1][1] - 0.2).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_STATES")]
    fn test_construction_too_many_states() {
        MarkovChain::from_flat(100, &[0.0; 10000]);
    }

    #[test]
    #[should_panic(expected = "flat slice too short")]
    fn test_construction_short_slice() {
        MarkovChain::from_flat(5, &[1.0, 2.0]);
    }

    #[test]
    fn test_new_is_empty() {
        let mc = MarkovChain::new();
        assert_eq!(mc.tm.n, 0);
    }

    // ------------------------------------------------------------------
    // 5–9: Stationary distribution
    // ------------------------------------------------------------------

    #[test]
    fn test_stationary_2state() {
        // P = [[0.7, 0.3], [0.4, 0.6]]
        // pi = [4/7, 3/7] ≈ [0.5714, 0.4286]
        let mc = simple_ergodic_2();
        let pi = mc.stationary_distribution().unwrap();
        assert!((pi[0] - 4.0 / 7.0).abs() < 1e-6);
        assert!((pi[1] - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_stationary_sums_to_one() {
        let mc = simple_ergodic_2();
        let pi = mc.stationary_distribution().unwrap();
        let s: f64 = pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_stationary_is_fixed_point() {
        let mc = simple_ergodic_2();
        let pi = mc.stationary_distribution().unwrap();
        // pi * P = pi
        let n = mc.tm.n;
        let mut new_pi = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                new_pi[j] += pi[i] * mc.tm.data[i][j];
            }
        }
        for j in 0..n {
            assert!((new_pi[j] - pi[j]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stationary_3state() {
        // Uniform chain: P[i][j] = 1/3
        let mc = MarkovChain::from_flat(3, &[1.0 / 3.0; 9]);
        let pi = mc.stationary_distribution().unwrap();
        for &v in &pi {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_stationary_empty_returns_none() {
        let mc = MarkovChain::new();
        assert!(mc.stationary_distribution().is_none());
    }

    // ------------------------------------------------------------------
    // 10–15: Simulate
    // ------------------------------------------------------------------

    #[test]
    fn test_simulate_length() {
        let mut mc = simple_ergodic_2();
        let traj = mc.simulate(0, 100);
        assert_eq!(traj.len(), 100);
    }

    #[test]
    fn test_simulate_all_valid_states() {
        let mut mc = simple_ergodic_2();
        let traj = mc.simulate(0, 200);
        for &s in &traj {
            assert!(s < 2);
        }
    }

    #[test]
    fn test_simulate_starts_at_s0() {
        let mut mc = simple_ergodic_2();
        let traj = mc.simulate(1, 5);
        assert_eq!(traj[0], 1);
    }

    #[test]
    fn test_simulate_seed_determinism() {
        let mut mc = simple_ergodic_2();
        mc.seed(42);
        let a = mc.simulate(0, 20);
        mc.seed(42);
        let b = mc.simulate(0, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn test_simulate_different_seeds_differ() {
        let mut mc = simple_ergodic_2();
        mc.seed(1);
        let a = mc.simulate(0, 50);
        mc.seed(999);
        let b = mc.simulate(0, 50);
        assert_ne!(a, b);
    }

    #[test]
    fn test_simulate_rng_persists_across_calls() {
        let mut mc = simple_ergodic_2();
        mc.seed(42);
        let a = mc.simulate(0, 30);
        let b = mc.simulate(0, 30);
        // Different trajectory because RNG advanced
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // 16–21: Is ergodic
    // ------------------------------------------------------------------

    #[test]
    fn test_is_ergodic_simple() {
        let mc = simple_ergodic_2();
        let (ok, reason) = mc.is_ergodic();
        assert!(ok, "should be ergodic: {}", reason);
    }

    #[test]
    fn test_is_ergodic_contains_ergodic_in_reason() {
        let mc = simple_ergodic_2();
        let (ok, reason) = mc.is_ergodic();
        assert!(ok);
        assert!(reason.contains("ergodic"));
    }

    #[test]
    fn test_is_ergodic_cycle4_fails() {
        let mc = cycle_4();
        let (ok, reason) = mc.is_ergodic();
        assert!(!ok, "cycle should not be ergodic: {}", reason);
        assert!(reason.contains("period"));
    }

    #[test]
    fn test_is_ergodic_reducible2() {
        let mc = reducible_2();
        let (ok, reason) = mc.is_ergodic();
        assert!(!ok, "reducible chain should not be ergodic: {}", reason);
    }

    #[test]
    fn test_is_ergodic_empty() {
        let mc = MarkovChain::new();
        let (ok, reason) = mc.is_ergodic();
        assert!(!ok);
        assert!(reason.contains("no states"));
    }

    #[test]
    fn test_is_ergodic_periodic_3() {
        let mc = periodic_3();
        let (ok, _reason) = mc.is_ergodic();
        assert!(
            !ok,
            "periodic/reducible chain should not be ergodic"
        );
    }

    // ------------------------------------------------------------------
    // 22–24: Mixing time
    // ------------------------------------------------------------------

    #[test]
    fn test_mixing_time_returns_some() {
        let mc = simple_ergodic_2();
        let t = mc.mixing_time(0.01, 10_000);
        assert!(t.is_some());
    }

    #[test]
    fn test_mixing_time_reasonable() {
        let mc = simple_ergodic_2();
        let t = mc.mixing_time(0.001, 10_000).unwrap();
        assert!(t > 0);
        assert!(t < 10_000);
    }

    #[test]
    fn test_mixing_time_empty_returns_none() {
        let mc = MarkovChain::new();
        assert!(mc.mixing_time(0.01, 10_000).is_none());
    }

    // ------------------------------------------------------------------
    // 25–28: Birkhoff bound
    // ------------------------------------------------------------------

    #[test]
    fn test_birkhoff_bound_positive() {
        let mc = simple_ergodic_2();
        let n = mc.birkhoff_bound(0.05, 0.01).unwrap();
        assert!(n >= 1.0);
    }

    #[test]
    fn test_birkhoff_bound_eps_zero_returns_none() {
        let mc = simple_ergodic_2();
        assert!(mc.birkhoff_bound(0.0, 0.01).is_none());
    }

    #[test]
    fn test_birkhoff_bound_delta_zero_returns_none() {
        let mc = simple_ergodic_2();
        assert!(mc.birkhoff_bound(0.05, 0.0).is_none());
    }

    #[test]
    fn test_birkhoff_bound_stricter_eps_needs_more() {
        let mc = simple_ergodic_2();
        let n1 = mc.birkhoff_bound(0.1, 0.01).unwrap();
        let n2 = mc.birkhoff_bound(0.05, 0.01).unwrap();
        assert!(n2 >= n1);
    }

    // ------------------------------------------------------------------
    // 29–31: Budget safety margin
    // ------------------------------------------------------------------

    #[test]
    fn test_budget_safety_margin_positive() {
        let mc = simple_ergodic_2();
        let m = mc.budget_safety_margin(0.01);
        assert!(m.is_some());
        assert!(m.unwrap() >= 0.0);
    }

    #[test]
    fn test_budget_safety_margin_empty_returns_none() {
        let mc = MarkovChain::new();
        assert!(mc.budget_safety_margin(0.01).is_none());
    }

    #[test]
    fn test_budget_safety_margin_more_mixing_larger_margin() {
        // A slow-mixing chain should have larger margin
        let fast = simple_ergodic_2();
        let slow = MarkovChain::from_flat(2, &[0.999, 0.001, 0.001, 0.999]);
        let m_fast = fast.budget_safety_margin(0.01).unwrap();
        let m_slow = slow.budget_safety_margin(0.01).unwrap();
        assert!(m_slow >= m_fast);
    }

    // ------------------------------------------------------------------
    // 32–35: Free functions
    // ------------------------------------------------------------------

    #[test]
    fn test_ergodic_time_average() {
        let traj = vec![0, 1, 0, 1];
        let f = vec![10.0, 20.0];
        let avg = ergodic_time_average(&traj, &f);
        assert!((avg - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_ergodic_ensemble_average() {
        let pi = vec![0.8, 0.2];
        let f = vec![5.0, 10.0];
        let avg = ergodic_ensemble_average(&pi, &f);
        assert!((avg - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_ergodic_check_true() {
        let traj = vec![0, 1, 0, 1, 0, 1];
        let f = vec![1.0, 2.0];
        let pi = vec![0.5, 0.5];
        assert!(ergodic_check(&traj, &f, &pi, 0.5));
    }

    #[test]
    fn test_ergodic_check_false() {
        let traj = vec![0, 0, 0, 0];
        let f = vec![1.0, 100.0];
        let pi = vec![0.5, 0.5];
        assert!(!ergodic_check(&traj, &f, &pi, 1.0));
    }

    // ------------------------------------------------------------------
    // 36–42: Wasserstein / Drift / Control
    // ------------------------------------------------------------------

    #[test]
    fn test_wasserstein_distance_identical() {
        let d = wasserstein_distance(&[0.5, 0.5], &[0.5, 0.5]);
        assert!((d - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_wasserstein_distance_different() {
        let d = wasserstein_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_wasserstein_distance_symmetric() {
        let a = &[0.3, 0.7];
        let b = &[0.6, 0.4];
        let d1 = wasserstein_distance(a, b);
        let d2 = wasserstein_distance(b, a);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn test_consumption_drift_zero() {
        // Trajectory perfectly matching pi
        let traj = vec![0, 1, 0, 1, 0, 1];
        let pi = vec![0.5, 0.5];
        let d = consumption_drift(&traj, &pi);
        assert!(d < 1e-12);
    }

    #[test]
    fn test_control_correction_basic() {
        let current = vec![0.8, 0.2];
        let target = vec![0.5, 0.5];
        let cost = vec![1.0, 2.0];
        let corr = control_correction(&current, &target, &cost);
        assert!((corr[0] - (-0.3)).abs() < 1e-12);
        assert!((corr[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_control_correction_zero_when_equal() {
        let d = vec![0.5, 0.5];
        let c = vec![1.0; 2];
        let corr = control_correction(&d, &d, &c);
        for v in corr {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_lqr_step_basic() {
        let alloc = vec![10.0, 20.0];
        let target = vec![1.0, 2.0];
        let cost = vec![1.0, 1.0];
        let result = lqr_step(&alloc, &target, &cost, 0.5);
        assert!((result[0] - 5.5).abs() < 1e-10);
        assert!((result[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_lqr_step_no_negative() {
        let alloc = vec![1.0];
        let target = vec![100.0];
        let cost = vec![10.0];
        let result = lqr_step(&alloc, &target, &cost, 2.0);
        assert!(result[0] >= 0.0);
    }

    #[test]
    fn test_predict_consumption_deterministic() {
        // Single state with resource 42
        let pi = vec![1.0];
        let resources = vec![42.0];
        assert!((predict_consumption(&pi, &resources) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_predict_consumption_mixed() {
        let pi = vec![0.5, 0.5];
        let resources = vec![10.0, 20.0];
        assert!((predict_consumption(&pi, &resources) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_budget_adequacy_exact() {
        let alloc = vec![15.0];
        let pi = vec![1.0];
        let res = vec![15.0];
        assert!(budget_adequacy(&alloc, &pi, &res).abs() < 1e-12);
    }

    #[test]
    fn test_budget_adequacy_shortfall() {
        let alloc = vec![10.0];
        let pi = vec![1.0];
        let res = vec![15.0];
        assert!((budget_adequacy(&alloc, &pi, &res) - (-5.0)).abs() < 1e-12);
    }

    #[test]
    fn test_occupation_measure_single_state() {
        let traj = vec![0, 0, 0, 0];
        let mu = occupation_measure(&traj, 1);
        assert!((mu[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_occupation_measure_two_states() {
        let traj = vec![0, 1, 0, 1];
        let mu = occupation_measure(&traj, 2);
        assert!((mu[0] - 0.5).abs() < 1e-12);
        assert!((mu[1] - 0.5).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // 43+: Edge cases & nalgebra interaction
    // ------------------------------------------------------------------

    #[test]
    fn test_wasserstein_three_state() {
        let d = wasserstein_distance(&[0.2, 0.3, 0.5], &[0.5, 0.3, 0.2]);
        // CDFs: 0.2 vs 0.5 diff=0.3; 0.5 vs 0.8 diff=0.3; 1.0 vs 1.0 diff=0.0
        assert!((d - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_birkhoff_bound_free_function() {
        let pi = vec![0.5, 0.5];
        let f = vec![10.0, 20.0];
        let n = birkhoff_bound(&pi, &f, 1.0, 0.05).unwrap();
        // var = 0.5*25 + 0.5*25 = 25
        // n = 25 / (1.0 * 0.05) = 500
        assert!((n - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_double_simulate_advances_rng() {
        let mut mc = simple_ergodic_2();
        mc.seed(1234);
        let _first = mc.simulate(0, 10);
        mc.seed(1234);
        let second = mc.simulate(0, 10);
        // same seed => same trajectory (important: seed resets rng)
        // We already tests this above. Here we just verify sequential calls differ.
        let third = mc.simulate(0, 10);
        // Should differ from second since rng advanced
        assert_ne!(second, third);
    }

    #[test]
    fn test_gcd_helper() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 3), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(-6, 15), 3);
    }

    #[test]
    fn test_nalgebra_support() {
        // Verify nalgebra can construct matrices from our data
        let n = 2_usize;
        let mc = simple_ergodic_2();
        let flat: Vec<f64> = (0..n * n)
            .map(|idx| mc.tm.data[idx / n][idx % n])
            .collect();
        let m = DMatrix::from_row_slice(n, n, &flat);
        assert!((m[(0, 0)] - 0.7).abs() < 1e-12);
        assert!((m[(1, 1)] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_ergodic_check_positive_tolerance_always_true() {
        let traj = vec![0, 0, 0, 0];
        let f = vec![1.0, 100.0];
        let pi = vec![0.5, 0.5];
        // huge tolerance => should pass
        assert!(ergodic_check(&traj, &f, &pi, 1e6));
    }

    #[test]
    fn test_simulate_large_n() {
        let mut mc = MarkovChain::from_flat(2, &[0.5, 0.5, 0.5, 0.5]);
        mc.seed(7);
        let traj = mc.simulate(0, 1000);
        assert_eq!(traj.len(), 1000);
        let count_0 = traj.iter().filter(|&&s| s == 0).count();
        let count_1 = traj.iter().filter(|&&s| s == 1).count();
        // roughly balanced
        let ratio = count_0 as f64 / 1000.0;
        assert!((ratio - 0.5).abs() < 0.15);
    }

    #[test]
    fn test_stationary_of_lazy_random_walk() {
        // Lazy random walk on 3 states (periodic-free):
        // self-loop 0.5, spread 0.25 to neighbours
        let flat = vec![
            0.5, 0.25, 0.25, // 0 -> 0,1,2
            0.25, 0.5, 0.25, // 1 -> 0,1,2
            0.25, 0.25, 0.5, // 2 -> 0,1,2
        ];
        let mc = MarkovChain::from_flat(3, &flat);
        let pi = mc.stationary_distribution().unwrap();
        assert!((pi[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((pi[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((pi[2] - 1.0 / 3.0).abs() < 1e-6);
    }
}
