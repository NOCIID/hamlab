"""
DeepHAM (Han, Yang & E, Quantitative Economics 2026) for the endogenous model:
    nets.py      policy and value networks with generalized moments of the distribution
    train.py     the algorithm: value regression, windowed policy loss with fictitious play
    agent.py     DeepHAMAgent (MFAX's agent interface; tracks the distribution in its hidden state)
    config.py    the options (the command line's arguments)
    __main__.py  the command line: python -m hamlab.solvers.deepham
"""
