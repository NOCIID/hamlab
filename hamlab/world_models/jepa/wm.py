"""
A JEPA world model of the economy.

The economy's endogenous state is the wealth distribution mu_t (1000 shares). Krusell and Smith
summarize it by capital K, and GPDP's law-of-motion GP (solvers/gpdp/lom.py) forecasts K. Here a joint-
embedding predictive architecture (JEPA) learns its own summary of mu, by prediction and without
reconstructing mu, as LeWM / LeWAM do for images:

    encoder    s_t = E(mu_t)                          d numbers
    predictor  s_{t+1} = s_t + P(s_t, z_t, tau_t)     tau_t = T - t, the time left
    loss       mean || predicted - encoded ||^2 over k-step rollouts  +  lambda * SIGReg(s)

Under a given population rule, mu_{t+1} is fixed by mu_t, today's shock z_t and the date (the
households' shares depend on them, and tomorrow's shock does not enter mu_{t+1}), so the predictor's
target is deterministic: the same structure as the K law of motion, with s in place of K. z is an
input of the predictor, not of the encoder, so s describes the distribution only.

SIGReg (LeJEPA; Balestriero and LeCun, 2025) prevents the collapse s = constant (which predicts
perfectly): it pushes random one-dimensional projections of the latents toward N(0, 1), through the
Epps-Pulley distance between their empirical characteristic function and the Gaussian's. There is
no stop-gradient on the targets (as in LeWM).

Hybrid (fit_latent_gp, latent_gp_step): the trained encoder is kept, and the MLP predictor is
replaced by a GP law of motion on the latent, as gpdp/lom.py does for K. The encoder learned a good
summary, but in experiments the MLP forecast K 50-80 times less accurately than the K law of motion.

Probe (fit_probe, probe_predict): a GP that reads capital K off the latent s, to measure what the
latent knows and to turn a forecast latent into a forecast of K.
"""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct


class Encoder(nn.Module):
    d: int
    width: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.Dense(self.width)(x))
        x = nn.gelu(nn.Dense(self.width // 2)(x))
        return nn.Dense(self.d)(x)


class Predictor(nn.Module):
    d: int
    width: int = 128

    @nn.compact
    def __call__(self, s, u):
        x = jnp.concatenate([s, u], axis=-1)
        x = nn.gelu(nn.Dense(self.width)(x))
        x = nn.gelu(nn.Dense(self.width)(x))
        return s + nn.Dense(self.d)(x)  # residual: predicts the change of the latent


@struct.dataclass
class WorldModel:
    enc_params: dict
    pred_params: dict
    d: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)


def encode(wm: WorldModel, mu):
    """Latent of distributions mu (..., n_states) -> (..., d). mu is scaled to mean 1 per entry."""
    return Encoder(wm.d).apply({"params": wm.enc_params}, mu * mu.shape[-1])


def exogenous(wm: WorldModel, z, time):
    """The predictor's other inputs: today's shock and the time left (scaled to [0, 1])."""
    return jnp.stack([z, (wm.T - time) / wm.T], axis=-1)


def predict(wm: WorldModel, s, z, time):
    """One step: s_t, z_t, t -> s_{t+1}."""
    return Predictor(wm.d).apply({"params": wm.pred_params}, s, exogenous(wm, z, time))


def rollout(wm: WorldModel, s0, z_seq, time_seq):
    """Open-loop latent path from s0 along given shocks and dates (k, ...): (k, ..., d)."""
    def _step(s, inputs):
        s = predict(wm, s, *inputs)
        return s, s

    return jax.lax.scan(_step, s0, (z_seq, time_seq))[1]


def sigreg(key, S, n_proj: int = 128, n_t: int = 17, t_max: float = 5.0):
    """
    SIGReg: mean over random unit directions u of the Epps-Pulley distance between the empirical
    characteristic function of S u (S: (n, d) latents) and that of N(0, 1), weighted by the Gaussian
    on a grid of t in [-t_max, t_max] (trapezoid rule). Zero when every projection is N(0, 1).
    """
    u = jax.random.normal(key, (S.shape[1], n_proj))
    proj = S @ (u / jnp.linalg.norm(u, axis=0))  # (n, n_proj)
    t = jnp.linspace(-t_max, t_max, n_t)
    arg = proj[..., None] * t  # (n, n_proj, n_t)
    gauss = jnp.exp(-0.5 * t**2)
    err = ((jnp.cos(arg).mean(0) - gauss) ** 2 + jnp.sin(arg).mean(0) ** 2) * gauss  # (n_proj, n_t)
    dt = t[1] - t[0]
    integral = dt * (err.sum(-1) - 0.5 * (err[:, 0] + err[:, -1]))
    return integral.mean()


def _loss(params, wm_static, mu, z, time, key, lam, n_proj):
    """k-step prediction loss on a batch of windows (mu: (B, k + 1, n), z, time: (B, k)) + SIGReg."""
    wm = WorldModel(params[0], params[1], *wm_static)
    S = encode(wm, mu)  # (B, k + 1, d)
    pred = rollout(wm, S[:, 0], jnp.swapaxes(z, 0, 1), jnp.swapaxes(time, 0, 1))  # (k, B, d)
    mse = jnp.mean((pred - jnp.swapaxes(S[:, 1:], 0, 1)) ** 2)
    reg = sigreg(key, S.reshape(-1, S.shape[-1]), n_proj)
    return mse + lam * reg, (mse, reg)


def train_world_model(seq, d: int = 4, k: int = 4, steps: int = 4000, batch: int = 512, lr: float = 1e-3,
                      lam: float = 0.1, n_proj: int = 128, seed: int = 0, log_every: int = 500):
    """
    Train the encoder and predictor on simulated paths seq (leading axes (T, E)), on random k-step
    windows (start period and economy drawn uniformly). Returns (WorldModel, log of
    (step, total loss, prediction mse, SIGReg)).
    """
    mu = jnp.asarray(seq.mu, jnp.float32)  # (T, E, n)
    z = jnp.asarray(seq.z, jnp.float32)
    time = jnp.asarray(seq.time, jnp.float32)
    T, E, n = mu.shape
    key = jax.random.PRNGKey(seed)
    k_enc, k_pred, key = jax.random.split(key, 3)
    enc_params = Encoder(d).init(k_enc, jnp.zeros((1, n)))["params"]
    pred_params = Predictor(d).init(k_pred, jnp.zeros((1, d)), jnp.zeros((1, 2)))["params"]
    params = (enc_params, pred_params)
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    static = (d, T)
    offsets = jnp.arange(k + 1)

    @jax.jit
    def _step(params, opt_state, mu, z, time, key):
        k_e, k_t, k_reg = jax.random.split(key, 3)
        e = jax.random.randint(k_e, (batch,), 0, E)
        t0 = jax.random.randint(k_t, (batch,), 0, T - k)
        rows = t0[:, None] + offsets  # (B, k + 1)
        mu_b = mu[rows, e[:, None]]  # (B, k + 1, n)
        z_b, time_b = z[rows[:, :k], e[:, None]], time[rows[:, :k], e[:, None]]
        (total, (mse, reg)), grads = jax.value_and_grad(_loss, has_aux=True)(
            params, static, mu_b, z_b, time_b, k_reg, lam, n_proj
        )
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, total, mse, reg

    log = []
    for i in range(steps):
        key, sub = jax.random.split(key)
        params, opt_state, total, mse, reg = _step(params, opt_state, mu, z, time, sub)
        if i % log_every == 0 or i == steps - 1:
            log.append((i, float(total), float(mse), float(reg)))
    return WorldModel(params[0], params[1], d, T), log


# --- hybrid: the JEPA's latent, with a GP law of motion on it instead of the MLP predictor ---
# As gpdp/lom.py's law of motion for K, but multi-output: inputs (s_t, z_t, log(T - t)), outputs
# s_{t+1} - s_t (d of them), one squared-exponential kernel shared by all outputs, fitted in float64
# (gpdp/lom.py: in float32 these nearly noiseless fits fail).
@dataclass
class LatentGP:
    X: np.ndarray  # (n, d + 2) standardized inputs
    alpha: np.ndarray  # (n, d)
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray  # (d,)
    y_std: np.ndarray
    log_ell: np.ndarray
    log_sf: float
    log_sn: float
    chol: np.ndarray = None  # (n, n) Cholesky factor of the Gram matrix, for predictive variances


def _latent_gp_inputs(S, z, time, T):
    return np.column_stack([np.asarray(S, np.float64), np.asarray(z, np.float64), np.log(T - np.asarray(time, np.float64))])


def fit_latent_gp(wm: WorldModel, seq, n_steps: int = 300, lr: float = 0.05):
    """Fit the latent law of motion on all transitions of simulated paths seq (leading axes (T, E))."""
    from jax.experimental import enable_x64

    from hamlab.solvers.gp_critic.critic import GPHyper, kernel
    from hamlab.solvers.gpdp.lom import LOM_LOG_ELL_BOUNDS, LOM_LOG_SF_BOUNDS, LOM_LOG_SN_BOUNDS

    S = np.asarray(encode(wm, jnp.asarray(seq.mu)), np.float64)  # (T, E, d)
    z, time = np.asarray(seq.z), np.asarray(seq.time)
    X = _latent_gp_inputs(S[:-1].reshape(-1, wm.d), z[:-1].reshape(-1), time[:-1].reshape(-1), wm.T)
    Y = (S[1:] - S[:-1]).reshape(-1, wm.d)
    x_mean, x_std = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    y_mean, y_std = Y.mean(0), np.maximum(Y.std(0), 1e-12)
    A, B = (X - x_mean) / x_std, (Y - y_mean) / y_std
    with enable_x64():
        A_, B_ = jnp.asarray(A), jnp.asarray(B)
        clip = lambda h: GPHyper(jnp.clip(h.log_ell, *LOM_LOG_ELL_BOUNDS), jnp.clip(h.log_sf, *LOM_LOG_SF_BOUNDS),  # noqa: E731
                                 jnp.clip(h.log_sn, *LOM_LOG_SN_BOUNDS))

        def nll(h):
            L = jnp.linalg.cholesky(kernel(A_, A_, h) + (jnp.exp(2 * h.log_sn) + 1e-10) * jnp.eye(A_.shape[0]))
            return (0.5 * jnp.sum(B_ * jax.scipy.linalg.cho_solve((L, True), B_)) / B_.shape[1]
                    + jnp.sum(jnp.log(jnp.diag(L)))) / A_.shape[0]

        @jax.jit
        def fit(h):
            tx = optax.adam(lr)
            grad = jax.value_and_grad(nll)

            def _step(carry, _):
                h, st, good = carry
                v, g = grad(h)
                u, new_st = tx.update(g, st)
                new = clip(optax.apply_updates(h, u))
                ok = jnp.isfinite(v) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(g)]))
                pick = lambda a, b: jax.tree.map(lambda p, q: jnp.where(ok, p, q), a, b)  # noqa: E731
                return (pick(new, h), pick(new_st, st), pick(h, good)), None

            h0 = clip(h)
            (h, _, good), _ = jax.lax.scan(_step, (h0, tx.init(h0), h0), None, length=n_steps)
            return jax.tree.map(lambda p, q: jnp.where(jnp.isfinite(nll(h)), p, q), h, good)

        h = fit(GPHyper(jnp.zeros(A.shape[1]), jnp.asarray(0.0), jnp.asarray(np.log(0.1))))
        L = jnp.linalg.cholesky(kernel(A_, A_, h) + (jnp.exp(2 * h.log_sn) + 1e-10) * jnp.eye(A_.shape[0]))
        alpha = np.asarray(jax.scipy.linalg.cho_solve((L, True), B_))
        return LatentGP(A, alpha, x_mean, x_std, y_mean, y_std, np.asarray(h.log_ell), float(h.log_sf), float(h.log_sn),
                        np.asarray(L))


def latent_gp_predict(gp: LatentGP, S, z, time, T: int = 128):
    """Mean and variance of s_{t+1} under the latent GP law of motion: S (m, d), z (m,), time (m,)
    -> (m, d), (m, d). The variance includes the GP's noise; all outputs share one kernel."""
    import scipy.linalg

    A = (_latent_gp_inputs(S, z, time, T) - gp.x_mean) / gp.x_std
    ell2, sf2 = np.exp(2 * gp.log_ell), np.exp(2 * gp.log_sf)
    k = sf2 * np.exp(-0.5 * (((A[:, None, :] - gp.X[None]) ** 2) / ell2).sum(-1))  # (m, n)
    mean = np.asarray(S, np.float64) + gp.y_mean + gp.y_std * (k @ gp.alpha)
    v = scipy.linalg.solve_triangular(gp.chol, k.T, lower=True)
    var_std = np.maximum(sf2 - np.sum(v**2, axis=0) + np.exp(2 * gp.log_sn), 0.0)  # (m,)
    return mean, var_std[:, None] * gp.y_std[None] ** 2


def latent_gp_step(gp: LatentGP, S, z, time, T: int = 128):
    """Mean of s_{t+1} under the latent GP law of motion: S (m, d), z (m,), time (m,) -> (m, d)."""
    A = (_latent_gp_inputs(S, z, time, T) - gp.x_mean) / gp.x_std
    ell2, sf2 = np.exp(2 * gp.log_ell), np.exp(2 * gp.log_sf)
    k = sf2 * np.exp(-0.5 * (((A[:, None, :] - gp.X[None]) ** 2) / ell2).sum(-1))
    return np.asarray(S, np.float64) + gp.y_mean + gp.y_std * (k @ gp.alpha)


# --- a precise probe s -> K: a single-output GP in float64, with the law of motion's bounds ---
# The policy-GP fit (gpdp.backward.fit_gp_bounded, float32, noise >= 1% of the output's spread) used
# as a probe has a noise floor of about 0.3% of K, comparable to the decode errors measured with it.
@dataclass
class Probe:
    X: np.ndarray  # (n, d) standardized latents
    alpha: np.ndarray  # (n,)
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    log_ell: np.ndarray
    log_sf: float
    log_sn: float


def fit_probe(S, y, n_steps: int = 300, lr: float = 0.05):
    """GP regression of y (n,) on latents S (n, d), in float64 (gpdp/lom.py's fit and bounds)."""
    from jax.experimental import enable_x64

    from hamlab.solvers.gp_critic.critic import GPHyper
    from hamlab.solvers.gpdp.lom import _fit_lom_hyper, _lom_gram

    S, y = np.asarray(S, np.float64), np.asarray(y, np.float64)
    x_mean, x_std = S.mean(0), np.where(S.std(0) > 1e-12, S.std(0), 1.0)
    y_mean, y_std = float(y.mean()), float(max(y.std(), 1e-12))
    A, b = (S - x_mean) / x_std, (y - y_mean) / y_std
    with enable_x64():
        hyp = GPHyper(jnp.zeros(S.shape[1]), jnp.asarray(0.0), jnp.asarray(np.log(0.1)))
        hyp, _ = _fit_lom_hyper(jnp.asarray(A), jnp.asarray(b), hyp, n_steps, lr)
        L = jnp.linalg.cholesky(_lom_gram(jnp.asarray(A), hyp))
        alpha = np.asarray(jax.scipy.linalg.cho_solve((L, True), jnp.asarray(b)))
        return Probe(A, alpha, x_mean, x_std, y_mean, y_std, np.asarray(hyp.log_ell), float(hyp.log_sf), float(hyp.log_sn))


def probe_predict(p: Probe, S):
    """Mean (m,) and gradient (m, d) of the probe at latents S (m, d)."""
    A = (np.asarray(S, np.float64) - p.x_mean) / p.x_std
    ell2, sf2 = np.exp(2 * p.log_ell), np.exp(2 * p.log_sf)
    diff = A[:, None, :] - p.X[None]  # (m, n, d)
    k = sf2 * np.exp(-0.5 * ((diff**2) / ell2).sum(-1))  # (m, n)
    mean = p.y_mean + p.y_std * (k @ p.alpha)
    grad_std = -np.einsum("mn,mnd->md", k * p.alpha[None], diff / ell2)  # d mean / d A
    return mean, p.y_std * grad_std / p.x_std


def probe_noise_var(p: Probe) -> float:
    return float(np.exp(2 * p.log_sn) * p.y_std**2)
