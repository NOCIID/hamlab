# GP critic for SPG / RSPG in MFAX (endogenous model)

This file documents adding a Gaussian-process critic to MFAX's structural policy gradient
(SPG) and its recurrent version (RSPG), on the endogenous (macro) model. It covers what was
added, why each choice was made, and every number measured while testing it. The same idea was
first implemented in the SRL tutorial code (github.com/StructuralRL/SRL), on the Huggett model.

The work went through two rounds:
1. **First round:** a direct port of the SRL critic, 2 seeds per method.
2. **Second round:** more seeds, plus fixes for what the first round revealed. The main fix is a
   time-aware critic.

The results below are final and multi-seed. Section 5 summarizes the first round and what changed.

## Results at a glance

**Recommended configuration (the default):** a time-aware GP critic with 32-period windows in 4
phases, which tile the 128-period episode.

```bash
python -m hamlab.solvers.gp_critic.rspg_gp        # or spg_gp
```

**RSPG, 5 seeds per method.** Exploitability is measured on 64 held-out economies; the training
time is for 500 gradient steps of 8 economies each, on an Apple M5 Pro CPU.

| | Original (128-period rollouts) | GP, time-aware (recommended) | GP, infinite-horizon version |
|---|---|---|---|
| Exploitability, mean ± sd | 0.039 ± 0.007 | **0.039 ± 0.005** | 0.045 ± 0.012 (4 seeds) + 1 collapsed (4.09) |
| ... as a consumption gain from deviating | 0.24% | 0.24% | 0.27% (4 seeds) |
| Permutation test vs original (p) | | **0.92** | 0.21 (with the collapse) |
| Training time | 1445 s | **304 s (4.8× less)** | 220 s (6.6× less) |
| Training time to exploitability ≤ 0.05 (median over seeds) | 1083 s | **276 s** | 179 s |
| Policy gap to original seed 0 (seed-to-seed noise: 2.7%) | 2.7% | **2.9%** | 4.1% (4 seeds) |
| Capital gap to original seed 0 (seed-to-seed noise: 7.2%) | 7.2% | **7.2%** | 12.6% (4 seeds) |
| Runs that collapsed | 0 / 5 | 0 / 5 | 1 / 5 |

**SPG, 3 seeds per method.**

| | Original | GP, time-aware | GP, infinite-horizon version |
|---|---|---|---|
| Exploitability | 0.048 ± 0.004 | **0.049 ± 0.002** (p = 0.7) | 0.055 ± 0.011 (p = 0.6) |
| Training time | 1391 s | **280 s (5.0× less)** | 204 s (6.8× less) |
| Training time to exploitability ≤ 0.07 | 830 s | **201 s** | 147 s |
| Policy gap / capital gap to original seed 0 (noise: 1.3% / 2.0%) | 1.3% / 2.0% | 1.4% / 2.0% | 1.4% / 1.6% |

**Findings:**
- **The time-aware GP version matches the original method.**
  - Its exploitability, policies and aggregate capital differ from the original by no more than
    the original differs from itself across seeds.
  - It does so with 4–5× less training time.
- **The time-aware critic is accurate at the end of training.** Against brute-force Monte Carlo
  of the true finite-horizon values, its mean error is 0.99% of consumption (RSPG) and 0.90%
  (SPG). The Monte Carlo noise is about 0.5%, and R² is 0.98 against a ceiling of 0.996.
- **The truncated gradient is the full gradient.** With the run's own trained critic, the
  time-aware gradient at the trained policy is within 3.7% of the full 128-period gradient, below
  the 4.3% Monte Carlo noise floor.
- **The infinite-horizon version, a direct port of the SRL critic, has a horizon bias.**
  - It estimates the value of a never-ending economy, while MFAX's episode ends at period 128.
  - Against the true finite-horizon values its error is 3.4%, and 7.8% at period 75.
  - Its gradient with its own critic is 7.9% off the full gradient, against a 1.8% noise floor.
  - In 4 of 5 seeds it saved somewhat more late in the episode.
  - The time-aware critic removes all of this (sections 2.5, 3.3, 4.2).
- **Lagged prices do not improve the critic,** in any variant. Current prices, equivalently
  (z, K), are enough.
- **Two weaknesses remain.**
  - *Collapse:* one seed of the infinite-horizon version collapsed. Its policy froze on a single consumption
    share with probability 1, so its gradient became zero (section 3.4). The cause is an
    uncapped policy distribution, which the original method shares. It did not happen in any
    original or time-aware run.
  - *Early critic error:* in the first 100–200 steps the critic is 5–30% off. It becomes accurate
    once the policy settles. Refitting it every step early on (fast refits) does not reduce this
    error reliably and does not change results, so fast refits are not the default (section 4.3).

---

## 1. What was added

No existing file was modified. The original `spg.py` and `rspg.py` stay as reference baselines.

| File | Role |
|---|---|
| `hamlab/solvers/gp_critic/critic.py` | The GP: ARD squared-exponential kernel shared by all outputs, marginal-likelihood fit (Adam), prediction, standardization; the aggregate features (z, K), the time left, and lags. |
| `hamlab/solvers/gp_critic/rollouts.py` | One policy interface for SPG and RSPG (`MFPolicyFn`); a step that never terminates (`inf_step`); memory-light rollouts with exact policy evaluation, finite or infinite horizon (`value_rollout`); GRU replay (`replay_hidden`). |
| `hamlab/solvers/gp_critic/train.py` | The method: the windowed loss with the GP tail, the staggered windows, the train step, the first critic, the refits (including the optional fast refits), checkpoints, logging and saving. Shared by SPG and RSPG. |
| `hamlab/solvers/gp_critic/spg_gp.py`, `rspg_gp.py` | Entry points (`python -m hamlab.solvers.gp_critic.rspg_gp`). Defaults: the time-aware critic, 32-period windows, phases (4, 2, 1, 1). |
| `configs/endogenous_spg_gp.yaml`, `configs/endogenous_rspg_gp.yaml` | W&B sweeps (learning rate) in the same format as the existing configs. |
| `experiments/gp_critic/check_equivalence.py` | Check 0: the new train step reproduces the original one exactly. |
| `experiments/gp_critic/check_gradient.py` | Check 1: the truncated gradient against the full-rollout gradient. |
| `experiments/gp_critic/check_critic.py` | Checks 2 and 3: the critic against brute-force Monte Carlo (finite or infinite horizon), the lag test, RSPG's memory use; also on checkpoints. |
| `experiments/gp_critic/compare_runs.py` | Check 4: exploitability on held-out economies, timing, policy and aggregate gaps. |
| `experiments/gp_critic/check_horizon.py` | Check 5: consumption and capital by period (end-of-episode behaviour). |
| `experiments/gp_critic/summarize_seeds.py` | Multi-seed statistics per method and an exact permutation test. |
| `experiments/gp_critic/run_all.sh` | Runs everything in sequence; reproduces every number in this file (about 9 hours on this CPU). |
| `experiments/gp_critic/results/` | JSON outputs and logs (`results/logs/`). `results/smoke/` holds small development tests, not cited here. |

Trained agents, critics, training curves and checkpoints are saved under
`runs/endogenous/hsm_{spg,rspg}_gp/<run_name>/` (`mf_agent_wrapper.pkl`, `gp_state.pkl`,
`log.json`, `args.json`, `ckpt_<step>/`). `runs/` is git-ignored.

### How to run

```bash
# Recommended: time-aware GP critic (defaults: H = 32, phases (4, 2, 1, 1), refit every 10 steps)
python -m hamlab.solvers.gp_critic.rspg_gp --run-name my_run

# Infinite-horizon critic, 25-period windows covering periods 0-99
python -m hamlab.solvers.gp_critic.rspg_gp --horizon 25 --no-time-input

# The original method through the same script (identical train step to rspg.py, see check 0)
python -m hamlab.solvers.gp_critic.rspg_gp --no-use-gp --horizon 128 --phase-counts 8

# Everything in this file (or selected stages: ONLY="stage|stage")
PY=python caffeinate -i bash experiments/gp_critic/run_all.sh
```

Environment: Python 3.10, `jax==0.5.3` (CPU), `flax==0.10.5`, `optax==0.2.3`, `chex==0.1.86`,
`distrax==0.1.5`, `gymnax==0.0.8`, `tyro==0.7.3`, `numpy==1.26.4`, `wandb==0.17.3`, on an Apple M5
Pro (18 cores, 48 GB). The environment was recreated once during the work. The new environment
reproduced earlier runs' numbers exactly.

---

## 2. The method

### 2.1 The objective

The original SPG/RSPG objective, for one gradient step, is the value of 8 economies over the full
128-period episode, weighted by the initial distribution:

```
J = mean_e  sum_x mu_0(x) V_0(x),     V_t = E_pi[r_t] + gamma * P_pi V_{t+1},    V_128 = 0
```

With the critic, each economy is simulated for H periods only (H = 32 by default), and the
backward recursion starts from the GP instead of zero:

```
V_H = stop_gradient( V_GP(x_H) ),     x_H = (z_H, K_H, T - t_H)  [+ lags]
```

In code this is one change to the scan's initial value (`make_loss` in `gp_critic/train.py`). The gradient
still flows through the policy's action probabilities inside `P_pi`: saving more moves mass towards
richer states, and `V_GP` says what that mass is worth. The critic is held fixed within a step.

### 2.2 The critic

- **Outputs:** the value of all 1000 individual states (200 wealth × 5 income points), as one vector.
  The individual state indexes the outputs, not the inputs, so the individual dynamics stay exact
  and the GP stays small.
- **Inputs:** `(z, K)`, the productivity shock and average wealth, plus the time left `T - t` in the
  time-aware version.
  - Average income L is constant in this model: the income transition is doubly stochastic, so
    the uniform marginal is stationary. Numerically, L = 0.841552 in every period.
  - The observed prices therefore reveal (z, K) exactly: `r / wage = alpha/(1-alpha) * L/K` gives
    K, then `r` gives z. A critic on (z, K) is a critic on what households observe, which is
    Krusell–Smith's approximate aggregation applied to the value function.
  - Optional lags add `(z_{t-1}, K_{t-1}), ...`.
- **GP:**
  - One ARD squared-exponential kernel, shared by all outputs.
  - Inputs and outputs are standardized, each output column separately.
  - The hyperparameters (log length scales, signal and noise std) are fitted by Adam on the
    negative log marginal likelihood, averaged over outputs, with jitter 1e-4.
  - The first fit takes 400 Adam steps; each refit takes 20 warm-started steps.
  - Lag length scales start at 10 (the no-lag model).
- **Inside `jit`:** the GP's arrays have fixed shapes (n = 1000), so the critic is passed through the
  train step's state like any other array. Unlike in the SRL notebook, no rewrite of the solver was
  needed to let the critic change.

### 2.3 Training targets

- **First critic:** 32 economies from the reset under the initial policy, evaluated by exact backward
  recursion.
  - Time-aware: whole 128-period episodes with the terminal reward; 4096 targets.
  - Infinite-horizon version: 256 periods with a step that never terminates and a zero tail; the first 128
    periods are kept, so each target misses at most 0.95^128 ≈ 0.14% of its value.
  - 1000 targets are subsampled; the fit takes 9–13 s.
- **Refits (every 10 steps):** the training windows already compute `V_t` at every period of every
  window, so those values are reused as targets: 10 steps × H periods × 8 economies, 1000
  subsampled. They cost nothing extra; the SRL notebook ran separate rollouts for its refits.
  Targets are bootstrapped from the previous critic at the end of each window.

### 2.4 Staggered windows

MFAX measures exploitability on 128-period episodes from the reset (uniform mu_0, z = 0),
discounted from period 0, and the original objective is weighted the same way.
- Restarting every window from the reset would train only the first H periods.
- Continuing economies indefinitely (the SRL scheme) would under-train the transient from the
  uniform distribution, which dominates the evaluation: with gamma = 0.95, 72% of the weight falls
  in the first 25 periods.

**The fix: cut each episode into windows of H periods and stagger them.**
- Economy slots are assigned to phases. Phase-0 slots start from the reset; a phase-k slot
  continues the economy that a phase-(k-1) slot simulated at the previous step.
- Each slot is weighted `gamma^(H k) / n_k`.
- Default: H = 32 with `phase_counts = (4, 2, 1, 1)`. Its 4 phases × 32 periods tile the whole
  episode. The infinite-horizon version used H = 25, covering periods 0–99.

**Why it is right.** The full-rollout gradient decomposes over windows (the actor-critic form of the
policy gradient): `grad J = sum_k gamma^(Hk) E[grad of window k]`. Window k's objective is its rewards
plus the critic's value of where it ends, weighted by the distribution at its start (the population,
which equals the individual distribution because both start from mu_0). So the staggered estimator
has the same expectation as the original gradient, up to the critic's error and the policy changing
by one step while an economy moves between phases. Check 1 tests this directly.

**RSPG's memory.** A phase-k slot needs the GRU state the policy would have after Hk periods.
- The slot stores the observations since the episode start. At each step the GRU state is
  recomputed from them with the current parameters (`replay_hidden`).
- This is cheap, avoids stale hidden states, and is done with gradients. The GRU weights therefore
  receive the same gradient as full backpropagation through time.

### 2.5 The time-aware critic

The infinite-horizon version's critic estimates the value of a never-ending economy: its targets come from a
step that hides the time counter from the environment, so the economy just continues past period
128. MFAX's episode, however, ends at period 128 with a terminal reward, and both the training
objective and the exploitability use that finite episode. As a result:
- the first critic counts utility after period 128 that never happens. Utility is negative here
  (u(c) = −1/c), so this makes wealth look more valuable than it is, increasingly towards the end
  of the episode. Against the true finite-horizon values its error is 3.4% on average and 7.8% at
  period 75 (check 2);
- with 25-period windows in four phases, periods 100–127 are never trained.

The fix (`--time-input --horizon 32 --phase-counts 4 2 1 1`, now the default):
- **The time left is a third input:** `x = (z, K, T - t)`. Its targets are finite-horizon values
  (`value_rollout(..., finite=True)` for the first critic; windows of the real episode for the
  refits).
- **The windows tile the episode.** The last phase's window ends at the true end: its last step
  pays the terminal reward, and its tail is zero.

### 2.6 Fast refits (optional, not the default)

`--fast-refit-steps 150 --fast-refit-window 5` refits the critic after every step for the first 150
steps, on the last 5 steps' targets, instead of every 10 steps. It was added to reduce the critic's
lag early in training. It does not reliably reduce the early critic error and does not change
results (sections 3.1, 4.3), and it costs about 45 s per run. It stays an option.

### 2.7 Equivalence with the original

With `--no-use-gp --horizon 128 --phase-counts 8`, the train step is the original one: same random
number stream, same rollout, same backward recursion, same loss (check 0). This configuration is the
"original" in every comparison, so all methods share the same script, logging, evaluation keys and
timing code.

---

## 3. Results

Common settings for all runs:
- 8 economies per step, 500 steps;
- learning rate 0.01, annealed to 0.001;
- gradient clipping at 1.0 (the defaults of `rspg.py`).

Exploitability during training uses a fixed set of 32 economies, the same for every run. The final
comparison uses 64 other, held-out economies (`compare_runs.py`). Seeds change the initial weights
and all random draws during training. Files: `results/compare_{rspg,spg}_all.json`,
`results/seeds_{rspg,spg}.json`, `results/horizon_all.json`.

### 3.1 RSPG: exploitability, time, gaps

Held-out exploitability by seed:

| Seed | 0 | 1 | 2 | 3 | 4 | Mean ± sd |
|---|---|---|---|---|---|---|
| Original | 0.033 | 0.040 | 0.034 | 0.051 | 0.037 | 0.039 ± 0.007 |
| GP time-aware | 0.046 | 0.041 | 0.039 | 0.031 | 0.040 | 0.039 ± 0.005 |
| GP time-aware + fast refits | 0.045 | 0.037 | 0.049 | 0.059 | 0.045 | 0.047 ± 0.008 |
| GP infinite-horizon version | 0.061 | 0.035 | 0.043 | 0.040 | **4.09 (collapsed)** | 0.045 ± 0.012 without seed 4 |

Exact permutation tests (5 vs 5) on held-out exploitability against the original: time-aware
p = 0.92; with fast refits p = 0.15; infinite-horizon version p = 0.21 (including the collapse).

Training time to reach a given exploitability (training evaluation set, evaluated every 50 steps,
median over the seeds that reach it):

| Threshold | Original | GP time-aware | GP + fast refits | GP infinite-horizon version |
|---|---|---|---|---|
| 0.10 | 864 s (5/5) | 215 s (5/5) | 227 s (5/5) | 149 s (4/5) |
| 0.07 | 1002 s (5/5) | 247 s (5/5) | 287 s (5/5) | 159 s (4/5) |
| 0.05 | 1083 s (4/5) | 276 s (5/5) | 317 s (5/5) | 179 s (4/5) |
| 0.042 | 1369 s (4/5) | 302 s (4/5) | 303 s (2/5) | 200 s (3/5) |

The time-aware version reaches each threshold in 3.9–4.5× less time. The infinite-horizon version is faster
still, because its windows are shorter (25 periods instead of 32). But it missed the thresholds in
the collapsed run, and it carries the horizon bias.

**Policies and aggregates.** Each method was compared with original seed 0:
- *Policies:* each policy's expected consumption share is evaluated along the same 16 episodes,
  population-weighted.
- *Aggregates:* each policy is run on its own 16 episodes with the same shocks, and capital is
  compared.

The original's own seeds 1–4 give the noise baseline.

| Method | Policy gap (mean) | Capital gap (mean) | Capital at period 100, mean (range) |
|---|---|---|---|
| Original (seeds 1–4 vs seed 0) | 2.7% | 7.2% | 28.0 (25.5–29.3), all 5 seeds |
| GP time-aware | 2.9% | 7.2% | 28.7 (27.6–30.0) |
| GP + fast refits | 2.9% | 6.8% | 28.2 (27.0–30.8) |
| GP infinite-horizon version (4 seeds) | 4.1% | 12.6% | 31.1 (28.4–34.8) |

The time-aware version is as close to the original as the original's seeds are to each other.

### 3.2 SPG

| Seed | 0 | 1 | 2 | Mean ± sd | Training time |
|---|---|---|---|---|---|
| Original | 0.045 | 0.053 | 0.047 | 0.048 ± 0.004 | 1391 s |
| GP time-aware | 0.048 | 0.052 | 0.047 | 0.049 ± 0.002 | 280 s |
| GP infinite-horizon version | 0.050 | 0.046 | 0.068 | 0.055 ± 0.011 | 204 s |

- Permutation p-values against the original: 0.7 (time-aware) and 0.6 (infinite-horizon version). With 3 vs
  3 seeds, the smallest possible p is 0.1.
- Time to reach exploitability 0.07: 830 s (original), 201 s (time-aware), 147 s (infinite-horizon version).
- Policy and capital gaps to original seed 0: 1.3% / 2.0% (original's own seeds), 1.4% / 2.0%
  (time-aware), 1.4% / 1.6% (infinite-horizon version).
- SPG has no memory, and all its variants agree closely.

### 3.3 End of the episode

Population-average consumption share (`check_horizon.py`, each policy on its own 16 episodes):

| | Periods 75–99 | Periods 123–127 | Runs that dissave before the end |
|---|---|---|---|
| RSPG original | 0.131 (4 seeds) / 0.145 (seed 0) | 0.131 / **0.165** (seed 0) | 1 / 5 |
| RSPG GP time-aware | 0.130 | 0.129 | 0 / 5 |
| RSPG GP + fast refits | 0.131 | 0.129 | 0 / 5 |
| RSPG GP infinite-horizon version (4 seeds) | 0.126 | 0.122 | 0 / 4 |
| SPG original / time-aware / infinite-horizon version | 0.130 / 0.128 / 0.128 | 0.126 / 0.123 / 0.123 | 0 |

- The infinite-horizon version saves more late in the episode: a 4% lower consumption share than the original
  seeds that do not dissave, and capital at period 100 averaging 31.1 against 28.0 (28.6 without the
  dissaving seed). The time-aware version does not: 0.130 against 0.131, and its capital (28.7) is
  within the original's range.
- One original RSPG seed (seed 0) learned to dissave before the end of the episode, which is the
  rational response to the terminal period. Its GRU acts as a clock: a zeroed memory changes actions
  about as much as the aggregate state does (check 3). Its consumption share rises from 0.143 to
  0.165, and capital falls from 26 to 21.5.
- The other 4 original seeds, and no GP run (time-aware included), learned this. With 1 of 5, it is
  a seed-dependent outcome rather than a property of the method.

### 3.4 Robustness: collapse

One run of the infinite-horizon version (RSPG, seed 4) collapsed:
- Between steps 100 and 150, its exploitability froze at exactly 4.084, and the policy return at
  −20.96.
- At every later evaluation, every state put probability 1.0000 on the same consumption share
  (12.5%), whatever its wealth, income or the prices.
- Rerunning the same seed with fast refits collapsed again, earlier (before step 50, frozen at
  10.02, on a 7.5% share).

**Mechanism.** The policy's Beta distribution has parameters `alpha, beta = 1 + softplus(.)`, with
no upper bound.
- If they become large, the distribution concentrates on one action bin, and the softmax over the
  20 bins becomes one-hot.
- Its gradient then vanishes, and training can never move it again.
- The original method shares this network.

**Trigger.** Unknown. It happens in the first ~150 steps, when the critic is still inaccurate, but
the fast-refit rerun shows that critic staleness alone is not the cause. One plausible
contributor: the first critic is fitted on the random initial policy, under which poor states look
extremely bad.

**Frequency.**

| Variant | Collapses |
|---|---|
| Infinite-horizon version (RSPG) | 1 / 5 |
| Original (RSPG and SPG) | 0 / 8 |
| Time-aware, with or without fast refits (RSPG and SPG) | 0 / 13 |
| Every other GP run (SPG infinite-horizon version, ablations, timing reruns) | 0 |

With these sample sizes, the true rates are uncertain: 1/5 is compatible with roughly 1–70%, and
0/5 with up to about 50% (95% intervals).

**Detection is easy.** The exploitability freezes at the same value across evaluations, and the
maximum action probability is 1 in every state.

**Possible guards** (not implemented):
- cap alpha and beta, or add an entropy bonus (these change the policy network, so the original
  method too);
- detect a collapse and restart with another seed.

### 3.5 Timing details

| | Original RSPG | GP time-aware | GP infinite-horizon version | Original SPG | SPG time-aware |
|---|---|---|---|---|---|
| Time per step (median) | 2.89 s | 0.57 s | 0.38 s | 2.78 s | 0.52 s |
| Refit (every 10 steps) | | 0.32 s | 0.31 s | | 0.32 s |
| First critic | | 9.5 s | 13 s | | 9.2 s |
| JIT compilation, first chunk | 30 s | 7 s | 6 s | 28 s | 6 s |

- The unmodified `rspg.py` and `spg.py` run at 2.87 s and 2.72 s per step, the same as the
  equivalent configuration in the new script.
- The time per step falls by 5.1× (time-aware) and 7.7× (infinite-horizon version), compared with
  128/32 = 4 and 128/25 = 5.1 fewer simulated periods.
- In the time-aware runs, the step time alternates: about 0.48 s in the chunk right after each
  evaluation, about 0.58 s in the other chunks. This is likely CPU frequency or thermal behaviour
  under sustained load; it is identical on mains and battery power. An earlier explanation, battery
  throttling, was wrong.
- Totals for 500 steps include the first critic, the refits and the pipeline fill, but exclude
  evaluation.
- One first-version run (GP seed 0) was slowed by the laptop sleeping. A clean rerun reproduced
  its curve exactly in 219.5 s.

---

## 4. Checks

### 4.0 Check 0: the new train step is the original one

`check_equivalence.py` runs the original `make_train_step` from `spg.py` / `rspg.py` and the new one
in its equivalent configuration from the same parameters and key:

| | SPG | RSPG |
|---|---|---|
| Random-number stream after 3 steps | identical | identical |
| Relative gradient difference | 6.5e-8 | 1.4e-7 |
| Max parameter difference after 3 Adam steps (each moves parameters by 0.01) | 3.0e-7 | 5.9e-6 |

- These are float32 rounding differences.
- Training is deterministic on CPU. An interrupted original run (seed 1) and two GP runs (seed 0)
  were rerun from scratch, and each rerun reproduced its exploitability curve exactly.

### 4.1 Check 1: the truncated gradient against the full gradient

`check_gradient.py` averages each gradient over many independent batches: 16 batches of 8 economies
for the full gradient, 64 for the windowed ones. It reports `||g − g_full|| / ||g_full||` and the
error expected from Monte Carlo noise alone, computed from split halves.

| Policy | Estimator (critic) | Error vs full | Noise floor | Norm ratio |
|---|---|---|---|---|
| initial | infinite-horizon version (fresh fit) | 0.99% | 0.41% | 0.991 |
| initial | zero tail (no critic) | 12.5% | 0.53% | 0.995 |
| initial | infinite-horizon version, reset-only windows | 35.8% | 0.43% | 0.675 |
| initial | **time-aware (fresh fit)** | **0.45%** | 0.83% | 0.997 |
| trained (infinite-horizon version, s0) | infinite-horizon version (fresh fit) | 2.2% | 1.5% | 1.013 |
| trained (infinite-horizon version, s0) | infinite-horizon version (**the run's own critic**) | 7.9% | 1.8% | 0.921 |
| trained (infinite-horizon version, s0) | zero tail | **439%** | 1.5% | 5.39 |
| trained (infinite-horizon version, s0) | reset-only windows | 25.2% | 2.1% | 0.748 |
| trained (time-aware, s0) | time-aware (fresh fit) | 12.0% | 5.6% | 1.120 |
| trained (time-aware, s0) | **time-aware (the run's own critic)** | **3.7%** | 4.3% | 1.024 |

- **With the critic it actually trains, the time-aware gradient is within its noise floor.** The
  fresh-fit check (12%) used a quick critic fitted by the check itself, from 32 short simulations.
  The infinite-horizon version's own critic gives 7.9% against a noise floor of 1.8%: its horizon bias shows
  in the gradient.
- **Without a critic, the truncated gradient is badly wrong.** At the trained policy it is 5.4× too
  large: once the policy is near-optimal, the missing future dominates, and the gradient pushes
  towards consuming more.
- **Reset-only windows point in the right direction but miss 25–36% of the gradient:** the
  contribution of later periods.
- **Cosines are not informative.** Every estimator has cosine > 0.98, because one direction (the
  overall saving rate) dominates. Use the relative error.

### 4.2 Check 2: the critic at the end of training

`check_critic.py` tests the critic where it is used: the window ends (H, 2H, 3H) of fresh episodes,
8 economies each. The reference value of each test state is the average of 32 independent
continuations, each evaluated by exact backward recursion:
- **"infinite":** 170 periods, never terminating;
- **"finite":** until period 128, with the terminal reward.

Errors are in consumption-equivalent (CE) units, as in the SRL work. With CRRA utility,
`V = K u(c_bar)`, so `c_bar_GP / c_bar_MC − 1 = (V_GP / V_MC)^(1/(1−sigma)) − 1`, where K cancels.
Utility here has no shift, so values never cross zero, and the plain relative value error is almost
identical. Errors are population-weighted means over the 1000 states and the 24 test states.

| Critic (run) | Reference | Mean error | By test period (signed) | Max visited | MC noise | R² (ceiling) |
|---|---|---|---|---|---|---|
| RSPG infinite-horizon version (gp_s0) | infinite | 0.72% | −0.2 / +0.6 / +0.7% | 2.8% | 0.47% | 0.960 (0.980) |
| RSPG infinite-horizon version (gp_s2) | infinite | 0.60% | −0.5 / −0.1 / −0.6% | — | 0.51% | 0.976 |
| RSPG infinite-horizon version (gp_s0) | **finite** | **3.4%** | −0.8 / −1.7 / **−7.8%** | — | 0.48% | 0.747 |
| **RSPG time-aware (gpfix_s0)** | finite | **0.99%** | −0.7 / +0.8 / +0.5% | 3.5% | 0.48% | 0.979 (0.996) |
| SPG infinite-horizon version (gp_s0) | infinite | 0.58% | — | 1.8% | 0.50% | 0.945 (0.969) |
| **SPG time-aware (gpfix_s0)** | finite | **0.90%** | −0.6 / +0.9 / +0.2% | 3.5% | 0.51% | 0.981 (0.996) |

The test periods are 25/50/75 for the infinite-horizon version and 32/64/96 for the time-aware one. The Monte
Carlo reference took 60–95 s per check.

### 4.3 Check 2c: the critic during training

`check_critic.py` was also run on checkpoints saved at steps 50, 100 and 250. Each checkpoint holds
the policy at that step and the critic in use, which was fitted on the previous 10 steps' rollouts.
Each critic is scored against its own reference.

| Critic error (CE) | Step 50 | Step 100 | Step 250 | Step 500 |
|---|---|---|---|---|
| Infinite-horizon version (RSPG gp_s2) | 11.5% | 15.4% | 0.9% | 0.6% |
| Time-aware (RSPG gpfix_s0) | 5.0% | 29% | 2.1% | 1.0% |
| Time-aware + fast refits (RSPG gpfast_s0) | 20% | 6.0% | | |

- **For the first 100–200 steps the critic is substantially off,** almost always too pessimistic:
  it values the improving policy as if it were still the worse, earlier one.
- **Fast refits are better at one checkpoint and worse at the other.** Staleness is therefore not
  the main cause. The likelier cause is bootstrapping: each refit's targets are built from the
  previous critic's predictions, so early errors carry forward until the policy settles.
- **Training converges anyway,** and the critic is accurate from about step 250 on.
- Early errors this large did not prevent the time-aware runs from matching the original. They may
  play a part in the one collapse.

Smaller development tests (60 steps) showed the same pattern: 46–66% at step 20, falling with
training (`results/smoke/`).

### 4.4 Check 2b: lags

For each check, GPs with 0, 1 and 2 lagged (z, K) were fitted on 800 *independent* targets: one per
simulated economy, from a single continuation. They were then scored on the Monte Carlo test set.
Each GP is fitted from two starting length scales (1 and 3), and the fit with the better marginal
likelihood is kept.

| Critic kind (run) | 0 lags: R² / error / NLML | 1 lag | 2 lags |
|---|---|---|---|
| RSPG infinite-horizon version (gp_s0) | 0.972 / 0.52% / 0.167 | 0.974 / 0.50% / 0.170 | 0.973 / 0.50% / 0.185 |
| RSPG time-aware (gpfix_s0) | 0.988 / 0.68% / −1.336 | 0.989 / 0.68% / −1.336 | 0.987 / 0.73% / −1.337 |
| SPG infinite-horizon version (gp_s0) | 0.952 / 0.55% / **0.257** | 0.575 / 1.32% / 0.454 | 0.595 / 1.42% / 0.460 |
| SPG time-aware (gpfix_s0) | 0.990 / 0.69% / −1.315 | 0.990 / 0.69% / −1.315 | 0.989 / 0.74% / −1.316 |

- Lags never improve accuracy, and the marginal likelihood never clearly prefers them.
- The lag length scales go long (4–13 in RSPG's first-version fits, 82–281 in its time-aware
  fits), which effectively switches the lags off.
- In SPG's infinite-horizon version, the lag GPs fall into a short-length-scale regime that fits noise. The
  marginal likelihood rejects them.
- **Correction to the first round:** it reported that lags make RSPG's critic worse (R² 0.873
  with 2 lags). That came from a fit stuck in a bad optimum (single start at length scale 1).
  With the better fit, lags neither help nor hurt.

### 4.5 Check 3: RSPG's memory

At the test states, the trained policy's expected consumption share was recomputed with altered
GRU states:

| Run | Zero memory | Another economy's memory | Variation across test states |
|---|---|---|---|
| RSPG infinite-horizon version (gp_s0) | 0.96% | 0.13% | 1.0% |
| RSPG time-aware (gpfix_s0) | 0.89% | 0.22% | 1.9% |

- The memory carries little economy-specific information: swapping in another economy's memory
  changes consumption by about 0.1–0.2%.
- A zeroed memory changes it about as much as moving between aggregate states. The GRU mostly
  encodes how far into the episode the economy is, which explains the dissaving seed (section 3.3).
- This fits with prices already revealing (z, K) exactly, and supports a critic that ignores the
  memory.

### 4.6 Ablations (first round, RSPG seed 0)

- **Zero tail** (25-period windows, `V_25 = 0`): exploitability 3.55 held-out, a 21.7% consumption
  gain from deviating. Households over-consume, and capital collapses from about 25 to about 6.
  The critic is what makes truncation work.
- **Reset-only windows with the GP:** exploitability 0.038, as good as the staggered version,
  although its gradient is 25–36% too small (check 1). The missing part mostly shrinks a correctly
  oriented gradient, and Adam normalizes step sizes. The staggered scheme is still the correct
  estimator and costs nothing extra.

---

## 5. The first round, and what changed

The first round ran the infinite-horizon version against the original with 2 seeds each (RSPG and SPG), plus
the ablations and checks 0–3.

**What it showed:**
- about 6.5× less training time;
- one GP seed matching the original;
- one GP seed ending above the original's range (0.061 vs 0.033–0.040);
- the GP runs saving more late in the episode;
- one original seed dissaving.

**What the second round changed:**
1. **More seeds:** 5 per method for RSPG, 3 for SPG, with permutation tests. GP seed 0's 0.061 turned
   out to sit near the original's own spread (0.033–0.051 over 5 seeds). But a fifth seed of the
   infinite-horizon version collapsed.
2. **The time-aware critic, with windows covering the whole episode** (section 2.5). It removed the
   horizon bias and the late-episode saving gap.
3. **Critic validation during training** (section 4.3), which found the early critic error.
4. **Fast refits** were tried against the early critic error. They neither reduced it reliably nor
   prevented the collapse.
5. **A more robust lag test** (section 4.4). It corrected the first round's "lags hurt" to "lags do
   not help".
6. **A gradient check with each run's own critic** (section 4.1). It resolved an apparent 12% gap
   for the time-aware version as an artefact of the check's own quick critic.

---

## 6. Observations about the model and the code

- **Prices reveal (z, K)** (section 2.2). The only hidden part of the aggregate state is the shape of
  the wealth distribution beyond its mean. This limits what RSPG's memory can add.
- **Interest rates.**
  - Under random initial policies, K is about 0.8 and r is about 0.35 per period.
  - Under trained policies, households consume 11–19% of cash-on-hand per period, K rises from
    16.5 to about 28, and r falls from 5.4% to about 3.2–4.4% per period.
  - An early remark that r ≈ 0.35 pointed to a calibration problem was wrong: that only holds
    under untrained policies.
- **No depreciation in production** (`r = e^z alpha (L/K)^(1-alpha)`): a modelling choice, worth
  stating.
- **The finite episode matters for the critic, not for exploitability.** The terminal period carries
  0.15% of the weight. But a critic that ignores the end is biased by several percent late in the
  episode (section 4.2), and RSPG's GRU can learn to anticipate the end (section 3.3). Whether
  end-of-episode dissaving is wanted depends on whether the 128-period episode is the model or a
  truncation of an infinite-horizon economy.
- **Evaluation noise.** `rspg.py` prints exploitability on 8 economies from a key drawn from the
  training key. Comparisons need a fixed key and more economies (`--eval-envs`, `--eval-seed`).
- **Exploitability is flat near the optimum.** Five original seeds span 0.033–0.051, and capital at
  period 100 varies by 15% between them. Differences between methods need several seeds to
  resolve.
- **The policy's Beta head is unbounded** (section 3.4). This is a robustness weakness of the
  existing network, not of the critic.

## 7. Differences from the SRL implementation (Huggett, notebook 7)

| | SRL notebook 7 | Here |
|---|---|---|
| Horizon | infinite (Huggett), so an infinite-horizon critic | finite 128-period episode, so a time-aware critic |
| Restarts | continue economies indefinitely after a burn-in | staggered windows tiling the evaluation episode; no burn-in |
| Refit targets | separate rollouts at each refit | the training windows' own values, reused |
| Changing critic inside `jit` | rewrote the solver's loop | the critic is passed as data in the train step's state |
| Recurrent policy | none (tabular) | GRU state recomputed from the episode's observations, with gradients |
| Critic inputs | (log z, q) | (z, K, time left), equivalently the observed prices plus time |
| Outputs | 600 states | 1000 states |
| Baseline | the authors' solver (different restart scheme) | the original train step, reproduced exactly |
| Seeds | 1 | 5 (RSPG), 3 (SPG), with permutation tests |
| Speedup | 2.5–2.7× | 4.8× (RSPG), 5.0× (SPG) in training time for the same number of steps |

## 8. Limitations and next steps

1. **Seeds.**
   - Five seeds per method (three for SPG) resolve large differences but not small ones. With
     p = 0.92, the time-aware and original exploitabilities are indistinguishable, but with a
     seed-to-seed sd of about 0.006, differences smaller than about 25–30% of the mean would likely
     go undetected.
   - Collapse rates are poorly estimated.
2. **Collapse** (section 3.4). Consider a cap on the Beta parameters or an entropy bonus. This
   affects the original method too, so test both.
3. **Early critic error** (section 4.3). Possible remedies:
   - fewer bootstrap generations early on (longer windows for the first steps);
   - critic targets from longer rollouts during warm-up;
   - a learning-rate warm-up.
4. **One configuration.** H = 32, one learning rate, refits every 10 steps, n = 1000. The sweep files
   vary the learning rate; window length and refit frequency remain to be explored.
5. **CPU only.** On a GPU, the original's 128-period rollout parallelizes better, so the speedup may be
   smaller. Measure there.
6. **The critic ignores RSPG's memory.** That is fine here, because the memory matters little
   (check 3). It may not be in a model where prices do not reveal the aggregate state.

## 9. Pitfalls met

- **Long batches get interrupted.** `run_all.sh` takes `ONLY="stage|..."` to run parts of the
  pipeline; interrupted runs restart from step 0 (training is deterministic).
- **Do not edit `run_all.sh` while it runs:** bash reads scripts as it executes them. Here all
  edits were after the running stage.
- **Laptop sleep** slowed a few chunks of one run. Prefer median chunk times
  (`step_time_median_s`), or rerun.
- **Environments in temporary folders** can be deleted by macOS's cleanup. A Python environment
  recreated with the same package versions reproduced earlier numbers exactly.
- **Cosine similarity hides gradient errors** when one direction dominates (check 1). Use the
  relative error.
- **Check design matters.** A gradient check with a freshly fitted critic understated the
  time-aware version (12% vs 3.7% with the run's own critic), and a single-start lag fit
  overstated the harm of lags.
