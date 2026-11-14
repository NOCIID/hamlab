<h1 align="center"> hamlab: heterogeneous-agent models and solvers in JAX </h1>

hamlab solves heterogeneous-agent macroeconomic models, in which a continuum of households saves and
consumes under idiosyncratic and aggregate risk, and prices depend on the whole wealth distribution.
Several solution methods run on the same models and are scored by the same evaluation:

- **GPDP**: Gaussian-process dynamic programming (backward induction with value and policy GPs);
- **DeepHAM** (Han, Yang & E, *Quantitative Economics* 2026): neural policy and value functions of
  generalized moments of the distribution;
- **SPG / RSPG with a GP critic**: structural policy gradients whose truncated rollouts are valued by a
  Gaussian process;
- **SPG / RSPG**: the structural policy gradient methods of [MFAX](https://github.com/CWibault/mfax),
  on which hamlab is built.

## Setup

Python 3.10, then, from the repository root:

```bash
pip install -r requirements.txt
```

## Running a solver

From the repository root:

```bash
python -m hamlab.solvers.gpdp --run-name my_gpdp --single-vgp --alpha 1.0   # one value GP, no blending
python -m hamlab.solvers.deepham --run-name my_deepham                      # the paper's settings
python -m hamlab.solvers.gp_critic.rspg_gp --run-name my_rspg               # RSPG with a GP critic
python -m hamlab.solvers.gp_critic.spg_gp --run-name my_spg                 # SPG with a GP critic
python -m mfax.algos.hsm.algos.rspg                                         # MFAX's own RSPG
```

Every solver takes `--model` (default `krusell_smith`) and lists its options with `--help`. Runs are
saved in `runs/endogenous/<algo>/<run name>/` (`args.json`, `log.json`, and the agent pickle).

## Models

A model holds everything about one economy: its parameters, the grid of individual states, the
budget constraint and utility, the idiosyncratic and aggregate shocks, the initial distribution and
how prices follow from the distribution. Solvers use a model only through the interface in
[`hamlab/models/base.py`](hamlab/models/base.py).

```python
from hamlab.models import make
model = make("krusell_smith")               # default parameters
model = make("krusell_smith", sigma=3.0)    # any parameter overridden
```

| Model | Economy |
|---|---|
| `krusell_smith` | households save in capital; Cobb–Douglas prices from average wealth and income; AR(1) productivity shock; 128 periods |
| `huggett` | households save in bonds in zero net supply; the bond price clears the market each period; Markov aggregate and idiosyncratic income; infinite horizon truncated at 170 periods; the calibration of Yang, Wang, Schaab and Moll's *Structural Reinforcement Learning* ([arXiv:2512.18892](https://arxiv.org/abs/2512.18892)) |

There are two kinds of model:
- **`Model`, explicit prices** (`krusell_smith`): prices are a formula of the distribution and the
  aggregate shock. Each such model is also an MFAX push-forward environment, so MFAX's algorithms
  (SPG, RSPG) and its exploitability run on it, as do hamlab's solvers.
- **`MarketClearingModel`** (`huggett`): the price is whatever clears a market, so it depends on what
  households choose at that price. The households' policy is given at every point of a grid of
  candidate prices; each period the model finds the clearing price (`clear`), then moves households
  at that price (`period`, `push_forward`). hamlab's solvers do not handle these models yet.

Each model is a folder with one file per economic block:

```
hamlab/models/
  base.py              the interface every model implements
  common/              blocks shared by models: grids.py, shocks.py (AR(1)), transitions.py
  krusell_smith/
    params.py          the calibration only (numbers and their meaning)
    households.py      cash on hand, saving, utility, terminal rule
    firms.py           production and factor prices
    markets.py         capital = average wealth, labour = average income
    shocks.py          aggregate productivity and idiosyncratic income moves
    model.py           the blocks assembled over time; the only file that knows about MFAX
  huggett/
    params.py          the calibration only
    households.py      cash on hand, consumption, bond purchases, utility; a closed-form starting policy
    markets.py         total savings at each candidate price, the clearing price
    shocks.py          aggregate income draws
    model.py           the blocks assembled over time (clear, period, push_forward, step)
```

The economics files contain the equations as plain functions of states and parameters, so each one
can be read and tested on its own. Each model reproduces its reference implementation exactly:
`krusell_smith` MFAX's `endogenous` environment (`experiments/models/check_krusell_smith.py`),
`huggett` the SRL tutorial code (`experiments/models/check_huggett.py`).

**Adding a model:** create `hamlab/models/<name>/` with the same blocks (`params.py` for the
calibration, one file per block of equations, `model.py` subclassing `Model` or
`MarketClearingModel`), reuse `common/` where it fits, and register the model in
`hamlab/models/__init__.py`.

## Evaluating

All solvers are scored by the same code, on 64 held-out economies:

```bash
python experiments/gpdp/step4_evaluate.py --runs runs/endogenous/gpdp/my_gpdp runs/endogenous/deepham/my_deepham
python experiments/gpdp/relative_errors.py --runs runs/endogenous/gpdp/my_gpdp runs/endogenous/deepham/my_deepham
```

- **Exploitability (MFAX):** the gain of a best response on the 20 grid actions that knows the
  realized aggregate shocks.
- **Consumption-equivalent (CE) relative errors:** each value is converted into the constant
  consumption giving the same lifetime utility, and the policy is compared with a best response
  using continuous shares, state by state and period by period, in % of consumption
  ([`hamlab/evaluation/metrics.py`](hamlab/evaluation/metrics.py)).

## Experiments

[experiments/](experiments/README.md) holds the checks, the batches that train and evaluate several
seeds, and the scripts that summarize them. Its README lists the runs each batch creates and the runs
each analysis script needs. The GP critic's method and results are written up in
[experiments/gp_critic/SUMMARY.md](experiments/gp_critic/SUMMARY.md).

## Repository layout

| Folder | Contents |
|---|---|
| `hamlab/core/continuous.py` | continuous consumption shares for the Krusell–Smith economy: dynamics, rewards, best responses and values along given paths; used by GPDP, DeepHAM and the evaluation |
| `hamlab/solvers/gpdp/` | GPDP: value GPs, a policy GP, an optional GP law of motion for capital, an optional policy GP with a learned memory |
| `hamlab/solvers/deepham/` | DeepHAM: generalized moments, a windowed policy loss with fictitious play, value regression |
| `hamlab/solvers/gp_critic/` | SPG / RSPG with a Gaussian-process critic |
| `hamlab/world_models/jepa/` | a JEPA world model of the wealth distribution, and its use as GPDP's law of motion |
| `hamlab/evaluation/` | consumption-equivalent accuracy measures shared by every solver, and loading of saved runs |
| `hamlab/models/` | the economic models |
| `experiments/` | checks, batches, summaries, and their saved results |
| `configs/` | Weights & Biases sweep files |
| `mfax/` | the MFAX library (see below) |
| `runs/` | trained runs (created by the solvers, not versioned) |

## Built on MFAX

[MFAX](https://github.com/CWibault/mfax) is a JAX library of mean-field games. hamlab uses its
push-forward environments (exact mean-field updates from the individual transition dynamics), its
structural policy gradient methods (SPG, RSPG) and its exploitability evaluation. `mfax/` also
contains MFAX's toy games (linear quadratic, beach bar) and its reinforcement-learning baselines
(M-OMD, IPPO, RIPPO). The dependency goes one way: hamlab imports MFAX, MFAX never imports hamlab.

If you use MFAX, please cite:

```bibtex
@misc{wibault2026recurrent,
      title={Recurrent Structural Policy Gradient for Partially Observable Mean Field Games},
      author={Clarisse Wibault and Sebastian Towers and Tiphaine Wibault and Juan Duque and Johannes Forkel and George Whittle and Andreas Schaab and Yucheng Yang and Chiyuan Wang and Michael Osborne and Benjamin Moll and Jakob Foerster},
      year={2026},
}
```

MFAX's M-OMD, IPPO and RIPPO build on [PureJaxRL](https://arxiv.org/abs/2210.05639)'s DQN, PPO and
Recurrent PPO implementations.
