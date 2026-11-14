"""
Policy GP with a learned memory.

A small GRU reads each economy's observed prices (r_t, w_t), and its hidden state h_t, after
observing period t, is an input of the policy GP: inputs (z_t, K_t, h_t), or h_t only (RSPG's
information set), and no date. The GRU's weights are fitted together with the GP's
hyperparameters by maximizing the GP's marginal likelihood of the backward pass's best shares: a
deep kernel with recurrent structure, as GP-LSTM (Al-Shedivat et al., JMLR 2017) and deep kernel
learning (Wilson et al., AISTATS 2016).

The difference from gpdp.policy.GRUFeatures (RSPG's GRU, frozen) is that this GRU is trained for
the policy GP. LearnedGRU has the same interface (step, hidden_size), so the forward pass
(gpdp.train.forward), RecurrentPolicyGP and GPDPRecurrentAgent (MFAX's exploitability) are reused
unchanged.

Against overfitting (Ober et al., UAI 2021, show that such kernels can overfit):
- a small GRU;
- the GP's noise floor (the bounds of gpdp.backward.fit_gp_bounded);
- weight decay on the GRU's weight matrices (not on its biases);
- early stopping on held-out stations, followed by a final solve on all stations;
- a warm start from the previous iteration's GRU.

"Chrono" initialization (Tallec & Ollivier, ICLR 2018): flax's GRU computes h' = (1 - z) n + z h.
With the update gate's bias at log(T_i - 1), z_i is about 1 - 1/T_i, and under steady inputs
h_i(t) ~ n_i (1 - z_i^t): a clock with time constant T_i. The time constants are spread from 2 to
128 periods, so the hidden state can encode the date before any training.

The fit runs in float64 (float32 factorizations of these kernel matrices are prone to NaNs); the result is
cast back to float32 for the forward pass.
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from jax.experimental import enable_x64
from jax.scipy.linalg import cho_solve

from hamlab.solvers.gpdp import backward as gb
from hamlab.solvers.gp_critic.critic import (
    GPHyper,
    GPState,
    _gram,
    aggregate_features,
    init_hyper,
    kernel,
    neg_log_marginal_likelihood,
)
from hamlab.solvers.gpdp.policy import SHARE_CLIP, RecurrentPolicyGP, logit, observed_prices


# --- the memory ---
@struct.dataclass
class LearnedGRU:
    """A GRU over standardized observed prices; same interface as gpdp.policy.GRUFeatures."""

    params: dict  # flax GRUCell parameters
    obs_mean: jax.Array  # (2,) standardization of the observed prices (set by fit_memory_policy)
    obs_std: jax.Array
    hidden_size: int = struct.field(pytree_node=False)
    # (low, high) bounds on the policy GP's predicted logits, read by gpdp/policy.py; None: no clipping
    # (and objects saved before this field existed read the class default, None, so they are unchanged)
    logit_bounds: tuple = struct.field(pytree_node=False, default=None)

    def step(self, hidden, obs, done):
        """One GRU update from the observed prices (E, 2): new hidden state (E, hidden_size)."""
        hidden = jnp.where(done[..., None], jnp.zeros_like(hidden), hidden)
        x = (obs - self.obs_mean) / self.obs_std
        new_hidden, _ = nn.GRUCell(self.hidden_size).apply({"params": self.params}, hidden, x)
        return new_hidden


def init_learned_gru(key, hidden_size: int = 8, chrono: bool = True, T: int = 128, n_obs: int = 2):
    """A fresh GRU (flax's default initialization), optionally with chrono biases (module docstring)."""
    params = nn.GRUCell(hidden_size).init(key, jnp.zeros((1, hidden_size)), jnp.zeros((1, n_obs)))["params"]
    params = jax.tree.map(lambda x: x, dict(params))
    if chrono:
        time_constants = jnp.geomspace(2.0, float(T), hidden_size)
        params["iz"] = {**params["iz"], "bias": jnp.log(time_constants - 1.0)}
        params["in"] = {**params["in"], "bias": jnp.ones(hidden_size)}  # n ~ tanh(1): the clocks move
    return LearnedGRU(params, jnp.zeros(n_obs), jnp.ones(n_obs), hidden_size)


def hidden_sequence(gru, obs_seq):
    """h_t after observing the prices of periods 0..t, as in the forward pass: (T, N, 2) -> (T, N, d)."""
    N = obs_seq.shape[1]
    done = jnp.zeros((N,), dtype=bool)

    def _step(h, obs):
        h = gru.step(h, obs, done)
        return h, h

    _, H = jax.lax.scan(_step, jnp.zeros((N, gru.hidden_size), obs_seq.dtype), obs_seq)
    return H


def path_obs(aggregate_s_seq):
    """Observed prices along paths with leading axes (T, N): (T, N, 2)."""
    return jax.vmap(observed_prices)(aggregate_s_seq)


# --- the joint fit ---
def _clip_hyper(hyp: GPHyper, log_sn_min, log_ell_min) -> GPHyper:
    """gpdp/backward.py's bounds, with a higher noise floor and a minimum length scale. With the
    policy GP's floors (noise 1% of the outputs' spread, length scales from 0.05), a memory GP can
    fit a short length scale with its noise at the floor, and overshoot between its training points
    (logits far outside the range of its targets)."""
    return GPHyper(
        log_ell=jnp.clip(hyp.log_ell, jnp.maximum(gb.LOG_ELL_BOUNDS[0], log_ell_min), gb.LOG_ELL_BOUNDS[1]),
        log_sf=jnp.clip(hyp.log_sf, *gb.LOG_SF_BOUNDS),
        log_sn=jnp.clip(hyp.log_sn, jnp.maximum(gb.LOG_SN_BOUNDS[0], log_sn_min), gb.LOG_SN_BOUNDS[1]),
    )


INPUT_LAYERS = ("ir", "iz", "in")  # flax GRUCell: the layers applied to the observed prices


def _decay(params, inputs_only: bool = False):
    """Sum of squares of the GRU's weight matrices (biases excluded: they hold the chrono clocks);
    inputs_only: only the matrices applied to the observed prices."""
    leaves = jax.tree_util.tree_leaves_with_path(params)
    return sum(
        jnp.sum(v**2)
        for p, v in leaves
        if getattr(p[-1], "key", None) == "kernel" and (not inputs_only or getattr(p[0], "key", None) in INPUT_LAYERS)
    )


def _inputs(gru_params, obs_mean, obs_std, hidden_size, obs, zk_std, with_zk):
    """Standardized policy-GP inputs at every station and decision period: (T-1, N, D)."""
    gru = LearnedGRU(gru_params, obs_mean, obs_std, hidden_size)
    H = hidden_sequence(gru, obs)[: zk_std.shape[0]]
    return jnp.concatenate([zk_std, H], axis=-1) if with_zk else H


@partial(jax.jit, static_argnames=("hidden_size", "with_zk", "n_steps", "n_val", "train_gru"))
def _fit(gru_params, hyp, obs_mean, obs_std, obs, zk_std, Y_fit, hidden_size, with_zk, n_steps, n_val,
         train_gru, lr, weight_decay, input_weight_decay, log_sn_min, log_ell_min):
    """
    Adam on (GRU weights, GP hyperparameters) for the NLML of the training stations (the first
    N - n_val), tracking the parameters with the smallest error on the held-out stations (the
    last n_val). A step from a non-finite loss is not taken. Returns (best params, best hyp, log).
    """
    N = obs.shape[1]
    n_tr = N - n_val
    Y_tr = Y_fit[:, :n_tr].reshape(-1, Y_fit.shape[-1])
    Y_va = Y_fit[:, n_tr:].reshape(-1, Y_fit.shape[-1])

    def features(theta):
        gru_params = theta[0] if train_gru else jax.lax.stop_gradient(theta[0])
        X = _inputs(gru_params, obs_mean, obs_std, hidden_size, obs, zk_std, with_zk)
        return X[:, :n_tr].reshape(-1, X.shape[-1]), X[:, n_tr:].reshape(-1, X.shape[-1])

    def loss(theta):
        X_tr, _ = features(theta)
        penalty = weight_decay * _decay(theta[0]) + input_weight_decay * _decay(theta[0], inputs_only=True)
        return neg_log_marginal_likelihood(theta[1], X_tr, Y_tr) + penalty * train_gru

    def val_error(theta):
        if n_val == 0:
            return jnp.asarray(0.0, Y_fit.dtype)
        X_tr, X_va = features(theta)
        L = jnp.linalg.cholesky(_gram(X_tr, theta[1]))
        pred = kernel(X_va, X_tr, theta[1]) @ cho_solve((L, True), Y_tr)
        return jnp.mean((pred - Y_va) ** 2)

    tx = optax.adam(lr)
    grad_fn = jax.value_and_grad(loss)

    def _step(carry, _):
        theta, opt_state, best, best_err = carry
        value, grads = grad_fn(theta)
        updates, new_state = tx.update(grads, opt_state)
        new = optax.apply_updates(theta, updates)
        new = (new[0], _clip_hyper(new[1], log_sn_min, log_ell_min))
        ok = jnp.isfinite(value) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)]))
        pick = lambda a, b: jax.tree.map(lambda u, v: jnp.where(ok, u, v), a, b)  # noqa: E731
        theta, opt_state = pick(new, theta), pick(new_state, opt_state)
        err = val_error(theta) if n_val > 0 else value
        better = jnp.isfinite(err) & (err < best_err)
        best = jax.tree.map(lambda u, v: jnp.where(better, u, v), theta, best)
        return (theta, opt_state, best, jnp.where(better, err, best_err)), (value, err)

    theta = (gru_params, _clip_hyper(hyp, log_sn_min, log_ell_min))
    init = (theta, tx.init(theta), theta, jnp.asarray(jnp.inf, Y_fit.dtype))
    (_, _, best, best_err), (losses, errs) = jax.lax.scan(_step, init, None, length=n_steps)
    return best[0], best[1], dict(loss=losses, val_error=errs, best_val_error=best_err)


def fit_memory_policy(
    env,
    stations,
    shares,
    gru: LearnedGRU,
    hyp=None,
    with_zk: bool = True,
    n_steps: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    n_val: int = 2,
    n_fit_cols: int = 125,
    train_gru: bool = True,
    set_obs_norm: bool = False,
    input_weight_decay: float = 0.0,
    noise_floor: float = 0.03,
    min_lengthscale: float = 0.05,
    clip_logits: bool = True,
):
    """
    Fit the GRU and the policy GP together on the backward pass's best shares.
    stations: aggregate states (T, N); shares: best shares (T-1, N, n_states), t = 0 .. T-2.
    gru: the starting GRU (warm start); set_obs_norm=True sets its price standardization from these
    stations (otherwise the GRU's is kept). Kept fixed from the first fit, it saturates the GRU when
    later economies' prices drift, so GPDP re-standardizes at every fit by default.
    input_weight_decay: extra penalty on the input weights only, pulling h toward the chrono clocks.
    noise_floor, min_lengthscale: lower bounds on the GP's noise and length scales (standardized
    outputs; inputs: standardized (z, K), and h in (-1, 1)), against overshooting (_clip_hyper).
    clip_logits: predictions are clipped to the range of the training targets (the clipped logits of
    gpdp.policy.SHARE_CLIP), so that an overshooting GP cannot give shares of 1e-10.
    The hyperparameters and the GRU are fitted on n_fit_cols output columns, on the first N - n_val
    stations, keeping the step with the smallest held-out error; the GP's weights are then solved
    for all outputs on all N stations. train_gru=False keeps the GRU fixed (a baseline).
    Returns (RecurrentPolicyGP with float32 arrays, info dict).
    """
    obs = np.asarray(path_obs(stations), np.float64)  # (T, N, 2)
    n_periods = shares.shape[0]
    obs_mean = obs.reshape(-1, 2).mean(0) if set_obs_norm else np.asarray(gru.obs_mean, np.float64)
    obs_std = obs.reshape(-1, 2).std(0) if set_obs_norm else np.asarray(gru.obs_std, np.float64)
    zk = np.asarray(jax.vmap(lambda a: aggregate_features(env, a))(stations), np.float64)[:n_periods]
    zk_mean, zk_std = zk.reshape(-1, 2).mean(0), zk.reshape(-1, 2).std(0)
    Y = np.asarray(logit(shares), np.float64)  # (T-1, N, J)
    y_mean, y_std = Y.reshape(-1, Y.shape[-1]).mean(0), np.maximum(Y.reshape(-1, Y.shape[-1]).std(0), 1e-6)
    Ys = (Y - y_mean) / y_std
    J = Y.shape[-1]
    cols = np.arange(0, J, max(1, J // n_fit_cols))[:n_fit_cols]
    d = gru.hidden_size
    D = d + 2 * with_zk
    if hyp is None:
        hyp = init_hyper(D, D)

    with enable_x64():
        f64 = lambda t: jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), t)  # noqa: E731
        params, hyp_fit, log = _fit(
            f64(gru.params), f64(hyp), jnp.asarray(obs_mean), jnp.asarray(obs_std), jnp.asarray(obs),
            jnp.asarray((zk - zk_mean) / zk_std), jnp.asarray(Ys[..., cols]), d, with_zk, n_steps, n_val,
            train_gru, lr, weight_decay, input_weight_decay, float(np.log(noise_floor)), float(np.log(min_lengthscale)),
        )
        # --- final GP: the chosen GRU and hyperparameters, weights for all outputs on all stations ---
        X = _inputs(params, jnp.asarray(obs_mean), jnp.asarray(obs_std), d, jnp.asarray(obs),
                    jnp.asarray((zk - zk_mean) / zk_std), with_zk).reshape(-1, D)
        L = jnp.linalg.cholesky(_gram(X, hyp_fit))
        alpha = cho_solve((L, True), jnp.asarray(Ys.reshape(-1, J)))
        nll = neg_log_marginal_likelihood(hyp_fit, X, jnp.asarray(Ys[..., cols].reshape(-1, len(cols))))
        x_mean = np.concatenate([zk_mean, np.zeros(d)]) if with_zk else np.zeros(d)
        x_std = np.concatenate([zk_std, np.ones(d)]) if with_zk else np.ones(d)
        f32 = lambda t: jax.tree.map(lambda x: jnp.asarray(np.asarray(x), jnp.float32), t)  # noqa: E731
        gp = GPState(f32(hyp_fit), f32(X), f32(alpha), f32(x_mean), f32(x_std), f32(y_mean), f32(y_std))
        bounds = (float(logit(SHARE_CLIP[0])), float(logit(SHARE_CLIP[1]))) if clip_logits else None
        new_gru = LearnedGRU(f32(params), f32(obs_mean), f32(obs_std), d, bounds)
        info = dict(
            nll=float(nll),
            loss_first=float(log["loss"][0]),
            loss_last=float(log["loss"][-1]),
            best_val_error=float(log["best_val_error"]),
            best_step=int(np.argmin(np.where(np.isfinite(np.asarray(log["val_error"])), np.asarray(log["val_error"]), np.inf))),
            finite=bool(np.all(np.isfinite(np.asarray(alpha)))),
        )
    return RecurrentPolicyGP(gp, new_gru), info
