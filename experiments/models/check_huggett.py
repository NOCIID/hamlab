"""
Check that hamlab's Huggett model reproduces the SRL tutorial code exactly (every comparison is exact
equality), against the reference outputs of make_huggett_reference.py (data/huggett_srl_reference.npz):

1. Grids: bonds, income and aggregate income chains, candidate prices; the truncation horizon.
2. Single periods, for the closed-form starting policy and random policies, uniform and random
   distributions, several aggregate states: total savings at every candidate price, the clearing
   price and its interpolation weights, each household's next states and their probabilities,
   utilities, the next aggregate state.
3. A whole economy of 170 periods under the starting policy, run as the SRL solver's simulation:
   aggregate states, clearing prices, distributions and discounted flow utilities in every period.

    python experiments/models/check_huggett.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hamlab.models import make as make_model  # noqa: E402
from hamlab.models.huggett import HuggettState  # noqa: E402
from hamlab.models.huggett import shocks  # noqa: E402

REF = Path(__file__).resolve().parent / "data" / "huggett_srl_reference.npz"
CASES = [("guess", None, 25, 0), ("guess", 101, 3, 1), ("guess", 102, 47, 2), (11, 103, 10, 3), (12, None, 40, 4)]
SIM_KEY = 7


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def report(name, ok):
    print(f"{'OK  ' if ok else 'FAIL'}  {name}")
    return ok


def main():
    ref = np.load(REF)
    model = make_model("huggett")
    p = model.params
    nb, ny, nq, nz, J = p.n_bonds, p.n_income, p.n_prices, p.n_z, model.n_states
    results = []

    # --- 1. grids ---
    results.append(report("grids: bonds, income chain, aggregate income chain, candidate prices", all([
        same(p.bond_grid, ref["b_grid"]), same(p.income_grid, ref["y_grid"]), same(p.income_transition, ref["y_trans"]),
        same(p.z_grid, ref["z_grid"]), same(p.z_transition, ref["z_trans"]), same(p.price_grid, ref["q_grid"]),
    ])))
    results.append(report(f"truncation horizon: {p.max_steps_in_episode} periods", p.max_steps_in_episode == int(ref["sim_T"])))

    # --- 2. single periods ---
    policies = {"guess": model.starting_policy()}
    for seed in (11, 12):
        policies[seed] = jax.random.uniform(jax.random.PRNGKey(seed), (nb, ny, nq, nz)).reshape(J, nq, nz)

    @jax.jit
    def one_period(choices, mu, z_idx, key):
        s = HuggettState(mu=mu, z_idx=z_idx, time=0)
        clearing = model.clear(s, choices)
        idxs, probs, u = model.period(s, choices, clearing)
        return clearing, idxs, probs, u * (1.0 - p.beta), shocks.next_aggregate_state(key, z_idx, p)

    for k, (pol, mu_seed, z_idx, key_seed) in enumerate(CASES):
        mu = jnp.ones(J) / J if mu_seed is None else jax.random.dirichlet(jax.random.PRNGKey(mu_seed), jnp.ones(J))
        clearing, idxs, probs, U, z_next = one_period(policies[pol][:, :, z_idx], mu, z_idx, jax.random.PRNGKey(key_seed))
        r = {name: ref[f"case{k}_{name}"] for name in ("rows", "cols", "vals", "U", "z_next", "S", "lo", "hi", "wl", "wh", "q")}
        ok = all([
            same(clearing.demand, r["S"]), same(clearing.lo, r["lo"]), same(clearing.hi, r["hi"]),
            same(clearing.w_lo, r["wl"]), same(clearing.w_hi, r["wh"]), same(clearing.price, r["q"]),
            same(np.broadcast_to(np.arange(J)[:, None], r["rows"].shape), r["rows"]),
            same(idxs, r["cols"]), same(probs, r["vals"]), same(U, r["U"]), same(z_next, r["z_next"]),
        ])
        label = "starting policy" if pol == "guess" else f"random policy {pol}"
        results.append(report(f"period {k}: {label}, {'uniform' if mu_seed is None else 'random'} distribution, "
                              f"z state {z_idx}: price {float(clearing.price):.6f} vs {float(r['q']):.6f}", ok))

    # --- 3. a whole economy, as the SRL solver simulates it ---
    policy = policies["guess"]

    @jax.jit
    def simulate(key):
        def step(carry, _):
            m, d, z_idx, discount, key = carry
            m_detached = jax.lax.stop_gradient(m)
            m_detached = m_detached / jnp.sum(m_detached, keepdims=True)
            key, subkey = jax.random.split(key)
            s = HuggettState(mu=m_detached, z_idx=z_idx, time=0)
            choices = policy[:, :, z_idx]
            clearing = model.clear(s, choices)
            idxs, probs, u = model.period(s, choices, clearing)
            flow = jnp.sum(u * (1.0 - p.beta) * d) * discount
            d, m = model.push_forward(d, idxs, probs), model.push_forward(m, idxs, probs)
            z_next = shocks.next_aggregate_state(subkey, z_idx, p)
            return (m, d, z_next, discount * p.beta, key), (z_idx, clearing.price, m, flow)

        key, subkey = jax.random.split(key)
        z0 = shocks.initial_aggregate_state(subkey, p)
        init = (jnp.ones(J) / J, jnp.ones(J) / J, z0, 1.0, key)
        _, path = jax.lax.scan(step, init, length=p.max_steps_in_episode)
        return path

    z_path, q_path, m_path, flow_path = simulate(jax.random.PRNGKey(SIM_KEY))
    results.append(report(
        f"economy of {p.max_steps_in_episode} periods: aggregate states, prices (from {float(q_path[0]):.4f} to "
        f"{float(q_path[-1]):.4f}), distributions, flow utilities",
        all([same(z_path, ref["sim_z"]), same(q_path, ref["sim_q"]), same(m_path, ref["sim_m"]), same(flow_path, ref["sim_flow"])]),
    ))

    print("ALL IDENTICAL" if all(results) else "SOME DIFFER")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
