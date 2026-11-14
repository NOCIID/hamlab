"""
The interface of an economic model.

A model is a heterogeneous-agent economy on a grid of individual states. It holds everything about
that economy: its parameters, the grid, the households' budget constraint and utility, the
idiosyncratic and aggregate shocks, the initial distribution, and how prices follow from the
distribution. hamlab's solvers and evaluation use a model only through this interface.

Every model is also an MFAX push-forward environment (mfax.envs.pushforward.base.PushforwardEnvironment):
it provides mf_reset, mf_step, mf_transition, mf_expected_value and mf_reward through the per-state
methods _single_pushforward_step and _single_pushforward_reward, so MFAX's algorithms (SPG, RSPG) and
its exploitability evaluation run on it unchanged.

Beyond MFAX's environment methods, a model provides:
    prices(mu, z)                     prices implied by a distribution and the aggregate shock
    prices_from_aggregates(z, K, L)   the same prices from capital K and labour L directly
    average_income(mu)                labour L under a distribution (batched)
    next_shock(z, eps)                tomorrow's aggregate shock for a standard normal innovation eps
    next_shock_moments(z)             mean and variance of tomorrow's shock (Gaussian shocks)
    summary(aggregate_s)              (z, K): the aggregate shock and capital, the low-dimensional
                                      description of the economy used by the GP-based solvers (batched)
    state_from_observation(mu, time, obs)
                                      the aggregate state of an economy whose distribution and date
                                      are known, from what households observe (batched)

This interface (Model) covers models whose prices are an explicit function of the distribution and
the aggregate shock (e.g. Krusell-Smith). Models whose price is set each period by market clearing
(e.g. Huggett) use MarketClearingModel below instead.

MarketClearingModel. When the price clears a market, it depends on what households choose at that
price, so households' choices cannot be fixed before the price is known (as an MFAX environment
requires). Instead, the households' policy is given at every point of a grid of candidate prices
(a demand schedule), and each period:
    clear(aggregate_s, choices)        the price at which the market clears, given the choices at
                                       every candidate price (no gradient flows through it: each
                                       household takes the price as given)
    period(aggregate_s, choices, c)    each household's choice at the clearing price, its utility,
                                       and its possible next states with their probabilities
    push_forward(mu, idxs, probs)      the next distribution
    step(key, aggregate_s, choices)    all of the above, plus the next aggregate shock
    reset(key)                         the initial aggregate state
"""

from abc import ABC, abstractmethod

from mfax.envs.pushforward.base import PushforwardEnvironment


class Model(PushforwardEnvironment):
    """An economy on a grid of individual states, with explicit prices (see the module docstring)."""

    name: str = ""

    @abstractmethod
    def prices(self, mu, z):
        """Prices for a distribution mu (n_states,) and aggregate shock z."""

    @abstractmethod
    def prices_from_aggregates(self, z, K, L):
        """Prices from the aggregate shock z, capital K and labour L (elementwise on batches)."""

    @abstractmethod
    def average_income(self, mu):
        """Labour L under distributions mu (..., n_states) -> (...)."""

    @abstractmethod
    def next_shock(self, z, eps):
        """Tomorrow's aggregate shock for today's z and a standard normal innovation eps."""

    @abstractmethod
    def next_shock_moments(self, z):
        """Mean and variance of tomorrow's aggregate shock given today's z (the shock is Gaussian)."""

    @abstractmethod
    def summary(self, aggregate_s):
        """(z, K) of an aggregate state: (2,), or (E, 2) for a batch."""

    @abstractmethod
    def state_from_observation(self, mu, time, obs):
        """Aggregate states (batch E) from known distributions mu (E, n_states), dates (E,) and the
        observed prices obs (E, obs_dim)."""


class MarketClearingModel(ABC):
    """An economy on a grid of individual states whose price is set each period by market clearing
    (see the module docstring)."""

    name: str = ""

    def __init__(self, params):
        self.params = params

    @property
    def n_states(self) -> int:
        return self.params.states.shape[0]

    @abstractmethod
    def reset(self, key, mu=None):
        """The initial aggregate state (the distribution mu, if given, instead of the model's own)."""

    @abstractmethod
    def clear(self, aggregate_s, choices):
        """Market clearing: the clearing price and its interpolation weights on the price grid."""

    @abstractmethod
    def period(self, aggregate_s, choices, clearing):
        """Households at the clearing price: (next-state indices, probabilities, utilities), per state."""

    @abstractmethod
    def push_forward(self, mu, idxs, probs):
        """The next distribution: mu moved along the households' transitions."""

    @abstractmethod
    def step(self, key, aggregate_s, choices):
        """One period: (next aggregate state, clearing, utilities)."""
