"""
Households: a household with wealth a and income level y has cash on hand m = (1 + r) a + w y, consumes
a share of it and saves the rest as next period's wealth. Utility is CRRA; in the last period it
consumes all its cash on hand.
"""


def cash_on_hand(wealth, income, interest_rate, wage):
    return (1.0 + interest_rate) * wealth + wage * income


def next_wealth(cash, share):
    """Savings: the part of cash on hand not consumed."""
    return cash * (1 - share)


def consumption(cash, share):
    return cash - cash * (1.0 - share)


def utility(c, sigma):
    """CRRA utility c^(1 - sigma) / (1 - sigma)."""
    return (c ** (1.0 - sigma)) / (1.0 - sigma)


def rewards(cash, share, sigma):
    """Utility of this period's consumption, and the terminal reward (consuming all cash on hand)."""
    return utility(consumption(cash, share), sigma), utility(cash, sigma)
