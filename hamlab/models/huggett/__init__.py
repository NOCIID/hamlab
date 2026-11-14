"""
The Huggett economy with aggregate risk: households save in bonds that are in zero net supply, and
the bond price is set each period by market clearing (Huggett 1993, with an aggregate income shock,
as in Yang, Wang, Schaab and Moll, "Structural Reinforcement Learning for Heterogeneous Agent
Macroeconomics", section 2.1).

    params.py      the calibration, and the grids derived from it
    households.py  cash on hand, consumption, bond purchases, utility
    markets.py     the bond market: total savings at each candidate price, the clearing price
    shocks.py      aggregate income (a Markov chain) and its draws
    model.py       the parts assembled over time, as a MarketClearingModel

The economy. An individual state is (bonds b, income level y), on a grid of 200 bond points
(geometric, dense near the borrowing limit -1, up to 50) times 3 income levels. A household receives
y z, where z is aggregate income, consumes and buys bonds at price q: c + q b' = b + y z. Log y and
log z follow AR(1) processes, discretized into Markov chains of 3 and 50 states (Tauchen). Utility is
CRRA with sigma = 2 and the discount factor is 0.96. Bonds are in zero net supply, so q is the price
at which households' total savings are zero; households' choices are given on a grid of 20 candidate
prices in [0.95, 0.99], and q is interpolated between them. The horizon is infinite, truncated after
170 periods (0.96^170 < 1e-3). Each economy starts from a uniform distribution over the grid and a
uniformly drawn aggregate income state.

Calibration and numerical choices are those of the paper (appendix A.1) and of its tutorial code
(github.com/StructuralRL/SRL, tutorials/05_ge_huggett_agg_risk.ipynb). For the same inputs and random
keys, this model produces exactly the same economies as that code (experiments/models/check_huggett.py).
"""

from hamlab.models.huggett.model import Clearing, Huggett, HuggettState
from hamlab.models.huggett.params import HuggettParams

__all__ = ["Clearing", "Huggett", "HuggettParams", "HuggettState"]
