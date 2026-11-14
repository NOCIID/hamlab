"""
DeepHAM's networks (Han, Yang & E, "DeepHAM: A global solution method for heterogeneous agent models
with aggregate shocks", Quantitative Economics 17(2), 2026; arXiv:2112.14377), for the endogenous
mean-field game. Training is in train.py.

Policy and value functions both take (individual state, aggregate shock z, moments of the
distribution) and, because MFAX's episode is finite (128 periods), the time left (the paper's Remark 2:
finite horizons need "only minor modification"). The moments are
  - fixed moments (n_fm = 1): mean wealth K, the paper's "1st moment" variant;
  - generalized moments (n_gm): Q = sum_j mu_j Q_NN(s_j), with Q_NN a small network (the "basis
    function", eq. (6) and (12)). The paper averages Q_NN over N = 50 simulated agents; here the
    distribution mu is exact, so the average is the continuum limit of the paper's (its footnote 2).
    As in the paper (footnote 10) and its code, Q_NN sees individual wealth only.
Policy and value functions have separate generalized moments (Q^C and Q^V, section 2.2).

Sizes and activations follow the reference code (github.com/frankhan91/DeepHAM, util.py and
configs/KS/game_nn_n50_0fm1gm.json): heads 24 x 24 tanh with a linear output, basis 12 x 12 tanh with
a linear output, Keras's default initialization (Glorot uniform weights, zero biases); the policy's
output goes through a sigmoid and is the consumption share of cash on hand (MFAX's action); the value
function is an ensemble of 3 networks whose (unnormalized) outputs are averaged. Inputs are
standardized with statistics of the first data set, then frozen (the reference's moving_average = 1.0).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from hamlab.solvers.gpdp.policy import SHARE_CLIP

# standardized input columns of Normalizer.x_mean / x_std
WEALTH, INCOME, Z, K, TIME_LEFT = range(5)


@dataclass(frozen=True)
class Config:
    """What the networks see (static)."""

    n_fm: int = 0  # fixed moments: 1 = mean wealth K
    n_gm: int = 1  # generalized moments
    use_time: bool = True  # time left T - t as an input
    gm_income: bool = False  # basis on (wealth, income) instead of wealth only
    T: int = 128  # episode length

    @property
    def n_inputs(self) -> int:
        return 3 + self.n_fm + self.n_gm + int(self.use_time)  # wealth, income, z, moments, time left


@struct.dataclass
class Normalizer:
    x_mean: jax.Array  # (5,) wealth, income, z, K, time left
    x_std: jax.Array
    v_mean: jax.Array  # value targets
    v_std: jax.Array


@struct.dataclass
class Net:
    """A policy or value function: the basis network of its generalized moments, and its head."""

    basis: list  # layers of Q_NN ([] when n_gm = 0)
    head: list


@struct.dataclass
class DeepHAMPolicy:
    """The population's policy: what deepham.train.forward and DeepHAMAgent evaluate."""

    net: Net
    norm: Normalizer
    cfg: Config = struct.field(pytree_node=False)


# --- a feedforward network as DeepHAM's util.FeedforwardModel ---
def init_mlp(key, sizes):
    """Layers [d_in, h_1, ..., d_out]: Glorot-uniform weights and zero biases (Keras's defaults)."""
    layers = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        lim = jnp.sqrt(6.0 / (a + b))
        layers.append(dict(w=jax.random.uniform(k, (a, b), minval=-lim, maxval=lim), b=jnp.zeros((b,))))
    return layers


def mlp(layers, x):
    """tanh hidden layers, linear output."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_net(key, cfg: Config, width=(24, 24), gm_width=(12, 12)) -> Net:
    k_basis, k_head = jax.random.split(key)
    basis = init_mlp(k_basis, [1 + cfg.gm_income, *gm_width, cfg.n_gm]) if cfg.n_gm else []
    return Net(basis=basis, head=init_mlp(k_head, [cfg.n_inputs, *width, 1]))


# --- inputs ---
def moments(env, cfg: Config, norm: Normalizer, basis, mu):
    """The moments of distributions mu (E, n_states), standardized K first: (E, n_fm + n_gm)."""
    states = env.params.states
    out = []
    if cfg.n_fm:
        k_mean = mu @ states[:, 0]
        out.append(((k_mean - norm.x_mean[K]) / norm.x_std[K])[:, None])
    if cfg.n_gm:
        xs = (states - norm.x_mean[:2]) / norm.x_std[:2]
        phi = mlp(basis, xs[:, : 1 + cfg.gm_income])  # Q_NN on every individual state: (n_states, n_gm)
        out.append(mu @ phi)  # the mu-weighted mean (the paper's (1/N) sum_i)
    return jnp.concatenate(out, axis=-1) if out else jnp.zeros((mu.shape[0], 0))


def inputs(env, cfg: Config, norm: Normalizer, basis, aggregate_s):
    """Inputs of a head for every individual state of every economy (leading axis E): (E, n_states, d)."""
    states = env.params.states
    xs = (states - norm.x_mean[:2]) / norm.x_std[:2]  # (n_states, 2)
    E, n = aggregate_s.mu.shape
    agg = [((aggregate_s.z - norm.x_mean[Z]) / norm.x_std[Z])[:, None], moments(env, cfg, norm, basis, aggregate_s.mu)]
    if cfg.use_time:
        tau = (cfg.T - aggregate_s.time).astype(xs.dtype)
        agg.append(((tau - norm.x_mean[TIME_LEFT]) / norm.x_std[TIME_LEFT])[:, None])
    agg = jnp.concatenate(agg, axis=-1)  # (E, d_agg)
    return jnp.concatenate(
        [jnp.broadcast_to(xs, (E, n, 2)), jnp.broadcast_to(agg[:, None, :], (E, n, agg.shape[-1]))], axis=-1
    )


# --- policy and value functions ---
def policy_shares(env, policy: DeepHAMPolicy, aggregate_s):
    """Consumption shares of every state for a batch of economies (leading axis E): (E, n_states)."""
    x = inputs(env, policy.cfg, policy.norm, policy.net.basis, aggregate_s)
    return jnp.clip(jax.nn.sigmoid(mlp(policy.net.head, x)[..., 0]), *SHARE_CLIP)


def value_normalized(env, cfg: Config, norm: Normalizer, net: Net, aggregate_s):
    """One value network's standardized output: (E, n_states)."""
    return mlp(net.head, inputs(env, cfg, norm, net.basis, aggregate_s))[..., 0]


def values(env, cfg: Config, norm: Normalizer, vnets: Net, aggregate_s):
    """The value ensemble (leaves with a leading axis num_vnet): mean of the unnormalized values, (E, n_states)."""
    out = jax.vmap(lambda net: value_normalized(env, cfg, norm, net, aggregate_s))(vnets)
    return out.mean(0) * norm.v_std + norm.v_mean
