# hamlab

Heterogeneous-agent models and solvers, built on [MFAX](https://github.com/CWibault/mfax).

`mfax/` (next to this folder) is the MFAX library, unchanged. It provides the environments (an
endogenous Krusell–Smith-style economy and two toy mean-field games), structural policy gradient
(SPG, RSPG) and RL baselines, and the exploitability evaluation. `hamlab/` extends it with other
solvers and with evaluation tools. The dependency goes one way: hamlab imports MFAX, MFAX never
imports hamlab.

## Layout

| Folder | Contents |
|---|---|
| `core/continuous.py` | continuous consumption shares for MFAX's endogenous environment: dynamics, rewards, best responses and values along given paths. Used by GPDP, DeepHAM and the evaluation. |
| `solvers/gpdp/` | Gaussian-process dynamic programming: backward induction with value GPs, a policy GP, an optional GP law of motion for capital, an optional policy GP with a learned memory |
| `solvers/deepham/` | DeepHAM (Han, Yang & E, *Quantitative Economics* 2026): policy and value networks with generalized moments of the distribution, a windowed policy loss with fictitious play, value regression |
| `solvers/gp_critic/` | SPG / RSPG with a Gaussian-process critic that values truncated rollouts ([method and results](../experiments/gp_critic/SUMMARY.md)) |
| `world_models/jepa/` | a JEPA world model of the wealth distribution, and its use as GPDP's law of motion |
| `evaluation/` | consumption-equivalent accuracy measures shared by every solver, and loading of saved runs |
| `models/` | economic models: `krusell_smith`, `huggett` (see [Models](#models)) |

Experiment scripts (checks, batches, summaries) are in [experiments/](../experiments).

## Running a solver

From the repository root:

```bash
python -m hamlab.solvers.gpdp --run-name my_gpdp --single-vgp --alpha 1.0   # one value GP, no blending
python -m hamlab.solvers.deepham --run-name my_deepham                      # the paper's settings
python -m hamlab.solvers.gp_critic.rspg_gp --run-name my_rspg               # RSPG with a GP critic
python -m hamlab.solvers.gp_critic.spg_gp --run-name my_spg                 # SPG with a GP critic
python -m mfax.algos.hsm.algos.rspg                                         # MFAX's own RSPG
```

Every solver takes `--model` (default `krusell_smith`) and lists its options with `--help`. Runs are saved in `runs/endogenous/<algo>/<run name>/`
(`args.json`, `log.json`, and the agent pickle).

## Models

A model holds everything about one economy: its parameters, the grid of individual states, the
budget constraint and utility, the idiosyncratic and aggregate shocks, the initial distribution and
how prices follow from the distribution. Solvers use a model only through the interface in
`models/base.py`.

```python
from hamlab.models import make
model = make("krusell_smith")               # default parameters
model = make("krusell_smith", sigma=3.0)    # any parameter overridden
```

| Model | Economy |
|---|---|
| `krusell_smith` | households save in capital; Cobb–Douglas prices from average wealth and income; AR(1) productivity shock; 128 periods (overview in `models/krusell_smith/__init__.py`) |
| `huggett` | households save in bonds in zero net supply; the bond price clears the market each period; Markov aggregate and idiosyncratic income; infinite horizon truncated at 170 periods; the calibration of Yang, Wang, Schaab and Moll's *Structural Reinforcement Learning* paper (overview in `models/huggett/__init__.py`) |

Two kinds of model (`models/base.py`):
- **`Model`, explicit prices** (`krusell_smith`): prices are a formula of the distribution and the
  aggregate shock. Each such model is also an MFAX push-forward environment, so MFAX's algorithms
  (SPG, RSPG) and its exploitability run on it, as do hamlab's solvers.
- **`MarketClearingModel`** (`huggett`): the price is whatever clears a market, so it depends on what
  households choose at that price. The households' policy is given at every point of a grid of
  candidate prices; each period the model finds the clearing price (`clear`), then moves households
  at that price (`period`, `push_forward`). hamlab's solvers do not handle these models yet.

Each model is a folder with one file per economic block:

```
models/
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
can be read and tested on its own.

Each model reproduces its reference implementation exactly: `krusell_smith` MFAX's `endogenous`
environment (`experiments/models/check_krusell_smith.py`), `huggett` the SRL tutorial code
(`experiments/models/check_huggett.py`, against reference outputs generated by
`make_huggett_reference.py`).

**Adding a model:** create `models/<name>/` with the same blocks (`params.py` for the calibration,
one file per block of equations, `model.py` subclassing `Model` or `MarketClearingModel`), reuse
`common/` where it fits, and register the model in `models/__init__.py`.

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
  (`evaluation/metrics.py`).

## Setup

Python 3.10 with `jax==0.5.3 jaxlib==0.5.3 flax==0.10.5 optax==0.2.3 chex==0.1.86 distrax==0.1.5
gymnax==0.0.8 tyro==0.7.3 numpy==1.26.4 wandb==0.17.3 scipy==1.15.2`.
