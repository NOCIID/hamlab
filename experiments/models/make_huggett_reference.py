"""
Reference outputs of the SRL tutorial code for the Huggett model with aggregate risk, used by
check_huggett.py to test hamlab.models.huggett.

The code below reproduces notebook 05 of github.com/StructuralRL/SRL (tutorials/05_ge_huggett_agg_risk.ipynb,
MIT licence): the grids, the closed-form initial policy, and AUS_func (one period: market clearing,
transitions A, utilities U, next aggregate state), with the helpers of its srl.utils package
(discretize.py, safe_linalg.py) loaded from a local copy of that repository. It saves:
  - the grids;
  - single periods (transitions, utilities, clearing, next aggregate state) for several policies,
    distributions and aggregate states;
  - a whole simulated economy (170 periods) under the closed-form policy, run exactly as the SRL
    solver's simulation does (srl/spg.py, SPGSolver._neg_simulated_utility).

    python -I experiments/models/make_huggett_reference.py --srl-repo /path/to/SRL
"""

import argparse
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "huggett_srl_reference.npz"
POLICY_SEEDS = (11, 12)  # random consumption-share policies, jax.random.uniform(PRNGKey(seed), (nb, ny, nq, nz))
CASES = [  # (policy: "guess" or a seed of POLICY_SEEDS, distribution seed or None for uniform, z_idx, key seed)
    ("guess", None, 25, 0),
    ("guess", 101, 3, 1),
    ("guess", 102, 47, 2),
    (11, 103, 10, 3),
    (12, None, 40, 4),
]
SIM_KEY = 7


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--srl-repo", required=True, help="a local copy of github.com/StructuralRL/SRL")
    cli = p.parse_args()
    utils = Path(cli.srl_repo) / "srl" / "utils"
    discretize = load(utils / "discretize.py", "srl_discretize")
    linalg = load(utils / "safe_linalg.py", "srl_safe_linalg")

    # --- notebook 05: calibration and grids ---
    nb, ny, nq, nz = 200, 3, 20, 50
    bc, eps = -1.0, 0.001
    gamma, sigma, B = 0.96, 2.0, 0.0
    b_grid = discretize.discrete_assets(0, 50 - bc, nb) + bc
    y_grid, y_trans = discretize.discrete_log_ar1(0.6, 0.2, ny)
    z_grid, z_trans = discretize.discrete_log_ar1(0.9, 0.02, nz)
    q_grid = jnp.linspace(0.95, 0.99, nq)
    b_dist = jnp.repeat(b_grid, ny)
    y_dist = jnp.tile(y_grid, nb)

    def gen_initial_guess():
        def per_state(b, y, q, z):
            w = b + y * z - bc
            p = 1.0 / q
            nom = jnp.maximum((p - (gamma * p) ** (1.0 / sigma)) * (w + 1.0 / (p - 1.0)), 0.0)
            return nom / w
        g = per_state(b_grid[:, None, None, None], y_grid[None, :, None, None],
                      q_grid[None, None, :, None], z_grid[None, None, None, :])
        return jnp.clip(g, 0.001, 1.0)

    def clearing(cpolicy, mt, z_idx):
        """The market-clearing block of AUS_func, unchanged, returning its intermediate results."""
        zt = z_grid[z_idx]
        J = nb * ny
        cshare_of_q = jax.lax.stop_gradient(cpolicy)[..., z_idx].reshape(-1, nq).T
        wealth_of_q = jnp.ones((nq, J)) * (b_dist + zt * y_dist)[None, :]
        C_of_q = jnp.clip((wealth_of_q - bc) * cshare_of_q, eps, wealth_of_q - (bc + eps) * q_grid[:, None])
        S_of_q = jnp.sum((wealth_of_q - C_of_q) * mt[None, :], axis=-1) / q_grid
        q_lo_i, q_hi_i, q_lo_w, q_hi_w = linalg.find_clearing_point(B, S_of_q)
        q = q_grid[q_lo_i] * q_lo_w + q_grid[q_hi_i] * q_hi_w
        return S_of_q, q_lo_i, q_hi_i, q_lo_w, q_hi_w, q

    def AUS_func(cpolicy, mt, st, key):
        """Notebook 05's AUS_func, with the policy given directly as consumption shares (the notebook
        parameterizes it by increments across the price grid and adds a training penalty in the first
        period; both are training devices, not part of the economy)."""
        z_idx, _ = st
        zt = z_grid[z_idx]
        J = nb * ny
        _, q_lo_i, q_hi_i, q_lo_w, q_hi_w, q = clearing(cpolicy, mt, z_idx)

        def build_triplets_for_state(idx_S):
            b_idx, y_idx = idx_S // ny, idx_S % ny
            b, y = b_grid[b_idx], y_grid[y_idx]
            cshare = (cpolicy[b_idx, y_idx, q_lo_i, z_idx] * q_lo_w + cpolicy[b_idx, y_idx, q_hi_i, z_idx] * q_hi_w)
            wealth = b + zt * y
            c = jnp.clip((wealth - bc) * cshare, eps, wealth - (bc + eps) * q)
            u = linalg.crra_util_func(c, sigma) * (1.0 - gamma)
            b_next = (wealth - c) / q
            b_lo, b_hi, wl, wh = linalg.interp_two_point_nonuniform(b_next, b_grid)
            cols = jnp.concatenate([b_lo * ny + jnp.arange(ny, dtype=jnp.int32), b_hi * ny + jnp.arange(ny, dtype=jnp.int32)])
            rows = jnp.full((2 * ny,), idx_S, dtype=jnp.int32)
            vals = jnp.concatenate([wl * y_trans[y_idx], wh * y_trans[y_idx]])
            return (rows, cols, vals), u

        A, U = jax.vmap(build_triplets_for_state)(jnp.arange(J, dtype=jnp.int32))
        z_idx_next = jax.random.choice(key, nz, p=z_trans[z_idx])
        return A, U, (z_idx_next, 0.0)

    def reset_func(key):
        return jax.random.choice(key, nz), 1.0

    policies = {"guess": gen_initial_guess()}
    for seed in POLICY_SEEDS:
        policies[seed] = jax.random.uniform(jax.random.PRNGKey(seed), (nb, ny, nq, nz))

    out = dict(b_grid=b_grid, y_grid=y_grid, y_trans=y_trans, z_grid=z_grid, z_trans=z_trans, q_grid=q_grid)

    # --- single periods ---
    aus = jax.jit(AUS_func)
    clear = jax.jit(clearing)
    J = nb * ny
    for k, (pol, mu_seed, z_idx, key_seed) in enumerate(CASES):
        mt = jnp.ones(J) / J if mu_seed is None else jax.random.dirichlet(jax.random.PRNGKey(mu_seed), jnp.ones(J))
        (rows, cols, vals), U, (z_next, _) = aus(policies[pol], mt, (z_idx, 0.0), jax.random.PRNGKey(key_seed))
        S, lo, hi, wl, wh, q = clear(policies[pol], mt, z_idx)
        out.update({f"case{k}_{name}": v for name, v in dict(
            rows=rows, cols=cols, vals=vals, U=U, z_next=z_next, S=S, lo=lo, hi=hi, wl=wl, wh=wh, q=q).items()})

    # --- a whole economy, as SRL's simulation (srl/spg.py) runs it, under the closed-form policy ---
    T = int(jnp.ceil(jnp.log(1e-3) / jnp.log(gamma)).astype(int))
    policy = policies["guess"]

    @jax.jit
    def simulate(key):
        def step(carry, _):
            m, d, s, discount, key = carry
            m_detached = jax.lax.stop_gradient(m)
            m_detached = m_detached / jnp.sum(m_detached, keepdims=True)
            key, subkey = jax.random.split(key)
            q = clear(policy, m_detached, s[0])[-1]
            A, U, s_next = AUS_func(policy, m_detached, s, subkey)
            flow_utility = jnp.sum(U * d) * discount
            d = linalg.apply_A_T(A, d)
            m = linalg.apply_A_T(A, m)
            return (m, d, s_next, discount * gamma, key), (s[0], q, m, flow_utility)

        key, subkey = jax.random.split(key)
        s0 = reset_func(subkey)
        init = (jnp.ones(J) / J, jnp.ones(J) / J, s0, 1.0, key)
        _, path = jax.lax.scan(step, init, length=T)
        return path

    z_path, q_path, m_path, flow_path = simulate(jax.random.PRNGKey(SIM_KEY))
    out.update(sim_z=z_path, sim_q=q_path, sim_m=m_path, sim_flow=flow_path, sim_T=T)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in out.items()})
    print(f"saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB): grids, {len(CASES)} single periods, a {T}-period economy")


if __name__ == "__main__":
    main()
