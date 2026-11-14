"""
Gaussian-process dynamic programming (GPDP) for the endogenous mean-field game: no gradient descent
and no policy network. Each iteration simulates the economy under the population's policy, computes
best responses backward with one value GP per period (averaging over tomorrow's shock, continuous
consumption shares), and fits a policy GP on a blend of the population's and the best shares.
The algorithm is in hamlab/solvers/gpdp/train.py, the options in config.py.

    python -m hamlab.solvers.gpdp --run-name my_run
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
from hamlab.solvers.gpdp.train import train  # noqa: E402
from hamlab.solvers.gpdp.config import args  # noqa: E402


if __name__ == "__main__":
    train(tyro.cli(args))
