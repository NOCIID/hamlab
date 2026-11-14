"""
Saved runs of any solver, for the held-out evaluation (experiments/gpdp/step4_evaluate.py,
relative_errors.py): what was saved, and the economy the run's own policy generates.
"""

import json
import pickle
from pathlib import Path

from hamlab.solvers.deepham import train as dht
from hamlab.solvers.gpdp import train as gt

# agent file of each kind of run, in the order they are looked for
AGENT_FILES = {"gpdp": "gpdp_agent.pkl", "deepham": "deepham_agent.pkl", "mfax": "mf_agent_wrapper.pkl"}


def load_run(run_dir):
    """
    A saved run: dict(name, args, agent, kind, continuous, log).
    kind: "gpdp", "deepham", or "mfax" (MFAX's own agents: SPG, RSPG and the GP-critic runs).
    continuous: the policy plays continuous consumption shares in its own economy (GPDP, unless it was
    trained on the grid actions, and DeepHAM); otherwise its economy is MFAX's.
    """
    run_dir = Path(run_dir)
    kind = next(k for k, f in AGENT_FILES.items() if (run_dir / f).exists())
    args = json.loads((run_dir / "args.json").read_text())
    with open(run_dir / AGENT_FILES[kind], "rb") as f:
        agent = pickle.load(f)
    log = json.loads((run_dir / "log.json").read_text()) if (run_dir / "log.json").exists() else {}
    continuous = kind == "deepham" or (kind == "gpdp" and not args.get("grid_actions", False))
    return dict(name=f"{run_dir.parent.name}/{run_dir.name}", args=args, agent=agent, kind=kind,
                continuous=continuous, log=log)


def continuous_economy(env, run, key, n_econ):
    """A continuous-share run's own economy: (aggregate states (T, E), shares (T, E, n_states))."""
    if run["kind"] == "deepham":
        return dht.forward(env, run["agent"].mf_params, key, n_econ)
    return gt.forward(env, run["agent"].mf_params, key, n_econ)[:2]
