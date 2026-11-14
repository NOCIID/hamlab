"""
DeepHAM (Han, Yang & E, "DeepHAM: A global solution method for heterogeneous agent models with
aggregate shocks", Quantitative Economics 17(2), 2026; arXiv:2112.14377) for the endogenous mean-field
game. Policy and value networks see (individual state, z, generalized moments of the distribution,
time left); the policy maximizes simulated utility over a window plus the value network's
continuation (fictitious play), the value networks regress on simulated lifetime utility.
See hamlab/solvers/deepham/train.py and nets.py.

    python -m hamlab.solvers.deepham --run-name my_run

The options, and how they map to the reference implementation's settings, are in config.py.
"""

import os
import sys
from pathlib import Path

import tyro

# --- make repo root importable even when file is executed by path ---
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "mfax").is_dir()), None
)
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# the solver (and with it jax) is imported before the options are built: with jaxlib 0.5.3, a garbage
# collection while jax is first imported can crash in its pytree registry
from hamlab.solvers.deepham.train import train  # noqa: E402
from hamlab.solvers.deepham.config import args  # noqa: E402


if __name__ == "__main__":
    train(tyro.cli(args))
