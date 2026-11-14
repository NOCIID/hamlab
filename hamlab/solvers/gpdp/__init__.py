"""
Gaussian-process dynamic programming (GPDP) for the endogenous model:
    backward.py      the backward pass: value GPs and best shares
    policy.py        the policy GP and GPDPAgent (MFAX's agent interface)
    lom.py           the GP law of motion for K and the closed-form expectation
    memory.py        a policy GP with a learned GRU memory
    memory_split.py  the SplitJEPA stage-1 diagnostic of that memory
    train.py         the GPDP loop (forward pass, backward pass, policy fit, evaluation)
    config.py        the options (the command line's arguments)
    __main__.py      the command line: python -m hamlab.solvers.gpdp
"""
