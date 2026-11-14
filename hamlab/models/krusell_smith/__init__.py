"""
A Krusell-Smith-style economy: households save in capital, and the interest rate and the wage come
from a Cobb-Douglas firm.

    params.py      the calibration, and the grids derived from it
    households.py  cash on hand, saving, utility, the terminal rule
    firms.py       production and factor prices
    markets.py     capital = average wealth, labour = average income
    shocks.py      aggregate productivity (AR(1)) and idiosyncratic income moves
    model.py       the parts assembled over time, as a hamlab Model and an MFAX environment

The economy. An individual state is (wealth a, income level y), on a grid of 200 wealth points
(geometric, 0 to 99) times 5 income levels (geometric, 0.1 to 2.0). A household consumes a share of
its cash on hand (1 + r) a + w y and saves the rest; utility is CRRA, and in the last of 128 periods it
consumes everything. Next-period wealth is split between its two neighbouring grid points (a lottery
with the same mean), and the income level moves one level up or down with probability 0.1 each.
Prices come from capital (average wealth), labour (average income) and productivity z, which follows
an AR(1) from z_0 = 0; the initial distribution is uniform over the grid. Households observe the
prices, not the distribution. The discount factor is set by the solvers (0.95 in the experiments).

Differences from the textbook model (Krusell and Smith 1998): 5 income levels instead of an
employment status, no unemployment insurance, a continuous AR(1) productivity shock instead of a
two-state chain, and a finite horizon.

This is MFAX's "endogenous" environment (mfax.envs.pushforward.macro.endogenous) organized by economic
block, with the same formulas and the same order of operations: for the same random keys and
actions it produces exactly the same economies (experiments/models/check_krusell_smith.py).
"""

from hamlab.models.krusell_smith.model import KrusellSmith, KrusellSmithState
from hamlab.models.krusell_smith.params import KrusellSmithParams

__all__ = ["KrusellSmith", "KrusellSmithParams", "KrusellSmithState"]
