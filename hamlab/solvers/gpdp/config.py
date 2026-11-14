"""
Options of the GPDP solver: the command line of `python -m hamlab.solvers.gpdp`.
"""

from dataclasses import dataclass


@dataclass
class args:
    # --- logging ---
    save: bool = True
    run_name: str = ""

    # --- environment ---
    task: str = "endogenous"
    model: str = "krusell_smith"  # the economic model (hamlab.models); `task` names the runs folder
    discount_factor: float = 0.95
    common_noise: bool = True

    # --- GPDP ---
    algo: str = "gpdp"
    seed: int = 0
    num_iterations: int = 30
    num_stations: int = 8  # economies simulated per iteration (the value GPs' training points per period)
    alpha: float = 0.5  # blending: policy <- (1 - alpha) * policy + alpha * best response
    start_share: float = 0.15  # starting policy: every state consumes this share of cash on hand
    n_nodes: int = 5  # Gauss-Hermite points for tomorrow's shock
    search_iter: int = 30  # golden-section iterations per best share
    policy_time: bool = True  # policy GP inputs (z, K, T - t); --no-policy-time: (z, K) only, without the date
    #                            (the value GPs stay one per period, as finite-horizon DP requires)
    policy_memory_run: str = ""  # RSPG run whose GRU (frozen) feeds the policy GP: inputs (z, K, hidden state);
    #                               use with --no-policy-time (RSPG's information set: no explicit date)
    policy_memory_zk: bool = True  # with memory: inputs (z, K, h); --no-policy-memory-zk: h only (as RSPG's network)
    policy_memory_learn: bool = False  # a GRU learned together with the policy GP (gpdp/memory.py), instead of
    #                                    RSPG's frozen one; use with --no-policy-time
    memory_hidden_size: int = 8  # learned GRU: hidden units
    memory_chrono_init: bool = True  # learned GRU: update-gate biases giving time constants from 2 to T periods
    memory_weight_decay: float = 1e-3  # learned GRU: penalty on its weight matrices
    memory_val_stations: int = 2  # learned GRU: stations held out for early stopping (the final solve uses all)
    memory_fit_cols: int = 125  # learned GRU: output columns used in the joint fit
    memory_steps_first: int = 300  # learned GRU: Adam steps, first fit
    memory_steps: int = 100  # learned GRU: Adam steps, later fits (warm started)
    memory_renorm_each_fit: bool = True  # learned GRU: re-standardize the observed prices at every fit (fixed
    #                                      at the first fit, it saturates the GRU when later prices drift)
    memory_input_weight_decay: float = 0.1  # learned GRU: extra penalty on its input weights, so that h stays close
    #                                         to the chrono clocks and prices only correct it (without it, h
    #                                         tracks price levels and generalizes poorly when prices shift)
    memory_noise_floor: float = 0.03  # learned GRU: lower bound on the policy GP's noise (standardized outputs)
    memory_min_lengthscale: float = 0.05  # learned GRU: lower bound on the policy GP's length scales (larger values
    #                                       keep the GP from resolving the slow clock units near the end of
    #                                       the episode; --memory-clip-logits prevents overshoots)
    memory_clip_logits: bool = True  # learned GRU: clip the policy GP's logits to its training targets' range
    grid_actions: bool = False  # train on MFAX's 20 grid actions (best responses among them; the
    #                             population plays its shares on the grid through MFAX's env.mf_step)
    law_of_motion: bool = False  # tomorrow's K from a GP law of motion (z, K, T - t) -> K' - K learned from
    #                              each forward pass; expectation over (z', K') in closed form (GPDP paper,
    #                              section 4 and Appendix A; gpdp/lom.py). With --single-vgp only.
    lom_model: str = "gp"  # with --law-of-motion: "gp", the K law of motion (gpdp/lom.py); "jepa", tomorrow's K from
    #                        a frozen JEPA encoder + a latent GP law of motion (jepa/lom.py; needs --jepa-model)
    jepa_model: str = ""  # path of a JepaModel pickle (experiments/gpdp/jepa_wm_pretrain.py)
    jepa_refit_probe: bool = False  # with --lom-model jepa: refit the probe (the GP s -> K) on each iteration's stations
    #                                 (only the encoder stays frozen)
    lom_support_paths: int = 0  # with --law-of-motion: 0 solves the Bellman equation at the stations' (z, K, t);
    #                             n > 0 at n support paths simulated with the law of motion itself (the
    #                             stations then only teach the law of motion; needs --alpha 1)

    # --- GPs (bounded hyperparameters, see gpdp.backward.fit_gp_bounded) ---
    gp_lr: float = 0.05
    vgp_steps_first: int = 300  # value GP of the last period, each iteration
    vgp_steps: int = 50  # value GPs of earlier periods (warm started from the next period's)
    single_vgp: bool = False  # ONE value GP over (z, K, T - t) for all periods instead of one per period
    vgp_hyp_steps: int = 30  # single value GP: Adam steps per re-fit (warm started)
    vgp_hyp_cols: int = 125  # single value GP: output columns used to fit the hyperparameters
    pgp_steps_first: int = 300  # policy GP, first iteration
    pgp_steps: int = 100  # policy GP, later iterations (warm started)
    lom_steps: int = 300  # law-of-motion GP, every iteration (cold started)

    # --- evaluation (as gp_critic/train.py: fixed economies and key) ---
    eval_frequency: int = 1
    eval_envs: int = 32
    eval_seed: int = 12345
