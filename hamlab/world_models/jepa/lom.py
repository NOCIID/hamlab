"""
The JEPA world model as GPDP's law of motion for K (used for forecasting only).

Decisions keep (z, K, T - t), which households can observe through prices: the JEPA latent is
computed from the whole wealth distribution, which households do not see. Only tomorrow's K in the
backward pass comes from the JEPA (the role of gpdp/lom.py's K law of motion):
    1. encode today's distribution:            s_t = E(mu_t)
    2. forecast the latent:                    s_{t+1} ~ N(m, diag(v)), the latent GP law of motion
                                               (jepa.wm.fit_latent_gp), refitted on each iteration's stations
    3. read K from the forecast latent:        K' mean = probe(m),
                                               K' variance = g^T diag(v) g + the probe's noise,
                                               g the gradient of the probe at m (first order)
The encoder and the probe (a GP s -> K) are trained once on economies covering a whole GPDP run
(experiments/gpdp/jepa_wm_pretrain.py) and frozen: like K, a fixed summary of the distribution.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hamlab.world_models.jepa import wm as jw
from hamlab.solvers.gp_critic.critic import gp_predict


@dataclass
class JepaModel:
    """The frozen parts: the world model's encoder (and its MLP predictor, unused here) and the probe s -> K."""

    wm: jw.WorldModel
    probe: object  # jepa.wm.Probe (float64 GP s -> K), or a gp_critic.critic.GPState (float32)


def save(model: JepaModel, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(model), f)


def load(path) -> JepaModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def encode_paths(model: JepaModel, seq):
    """Latents along simulated paths (leading axes (T, E)): (T, E, d), float64."""
    return np.asarray(jw.encode(model.wm, jnp.asarray(seq.mu)), np.float64)


def fit_latent_law(model: JepaModel, stations) -> jw.LatentGP:
    """This iteration's latent law of motion, fitted on the stations' transitions."""
    return jw.fit_latent_gp(model.wm, stations)


def _probe_mean_and_grad(probe, S):
    if isinstance(probe, jw.Probe):
        return jw.probe_predict(probe, S)
    f = lambda s: gp_predict(probe, s[None])[0, 0]  # noqa: E731
    S = jnp.asarray(S, jnp.float32)
    return np.asarray(jax.vmap(f)(S), np.float64), np.asarray(jax.vmap(jax.grad(f))(S), np.float64)


def _probe_noise(probe) -> float:
    if isinstance(probe, jw.Probe):
        return jw.probe_noise_var(probe)
    return float(np.exp(2 * np.asarray(probe.hyp.log_sn)) * np.asarray(probe.y_std)[0] ** 2)


def next_K(model: JepaModel, lgp: jw.LatentGP, S_t, z, time):
    """Mean and variance of K_{t+1} for economies with latents S_t (m, d), shocks z (m,) at period
    `time` (scalar or (m,)): forecast the latent, then read K from it (module docstring)."""
    S_t = np.asarray(S_t, np.float64)
    time = np.broadcast_to(np.asarray(time, np.float64), (S_t.shape[0],))
    m, v = jw.latent_gp_predict(lgp, S_t, np.asarray(z, np.float64), time, model.wm.T)
    K_mean, g = _probe_mean_and_grad(model.probe, m)
    return K_mean, np.sum(g**2 * v, axis=-1) + _probe_noise(model.probe)


def refit_probe(model: JepaModel, S, K) -> JepaModel:
    """
    The same frozen encoder with a probe s -> K refitted on this iteration's stations: latents S
    (T, N, d) and their capital K (T, N), known at the stations. With the probe frozen, economies
    outside its training range can make it extrapolate, and K' then goes badly wrong.
    """
    d = S.shape[-1]
    return JepaModel(model.wm, jw.fit_probe(np.asarray(S).reshape(-1, d), np.asarray(K).reshape(-1)))
