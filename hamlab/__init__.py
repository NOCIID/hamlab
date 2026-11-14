"""
hamlab: heterogeneous-agent models and solvers, built on MFAX.

mfax/ (next to this folder) is the MFAX library, unchanged: the environments, SPG / RSPG and the RL
baselines, and the exploitability evaluation. hamlab extends it:
    core/          continuous consumption shares: dynamics, best responses and values along paths
    solvers/       gpdp/ (Gaussian-process dynamic programming), deepham/ (DeepHAM),
                   gp_critic/ (SPG / RSPG with a Gaussian-process critic)
    world_models/  jepa/ (a JEPA world model of the economy, and its use as GPDP's law of motion)
    evaluation/    accuracy measures shared by every solver, and loading of saved runs
    models/        economic models: krusell_smith (explicit prices; also an MFAX environment) and
                   huggett (the bond price set by market clearing)
See hamlab/README.md.
"""
