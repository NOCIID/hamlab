"""
Options of the DeepHAM solver: the command line of `python -m hamlab.solvers.deepham`.

Settings follow the reference code's configs/KS/game_nn_n50_0fm1gm.json (github.com/frankhan91/DeepHAM).
Where MFAX's model differs (finite horizon of 128 periods, discount 0.95, the exact distribution
instead of 50 agents), the setting is translated as noted.
"""

from dataclasses import dataclass


@dataclass
class args:
    # --- logging ---
    save: bool = True
    run_name: str = ""

    # --- environment (as GPDP) ---
    task: str = "endogenous"
    model: str = "krusell_smith"  # the economic model (hamlab.models); `task` names the runs folder
    discount_factor: float = 0.95
    common_noise: bool = True

    # --- what the networks see (paper section 2.2) ---
    algo: str = "deepham"
    seed: int = 0
    n_gm: int = 1  # generalized moments (reference configuration "0fm1gm": 1)
    n_fm: int = 0  # fixed moments: 1 adds mean wealth K (the reference's main configuration: "n_fm": 1, "n_gm": 0)
    gm_income: bool = False  # basis on wealth only, as the paper (footnote 10) and code
    use_time: bool = True  # time left T - t as an input, for the finite horizon (the paper's Remark 2)
    allow_no_moments: bool = False  # permit n_fm = n_gm = 0 (an ablation: the policy sees (s, z, time) only)

    # --- networks (reference: policy/value net_width [24, 24], gm_config net_width [12, 12], tanh, num_vnet 3) ---
    net_width: tuple[int, ...] = (24, 24)
    gm_width: tuple[int, ...] = (12, 12)
    num_vnet: int = 3

    # --- start (reference: init_const_share 0.4; here the same starting share as GPDP) ---
    start_share: float = 0.15

    # --- value function (reference: value_config) ---
    value_paths: int = 36  # economies per value data set: 36 x 128 = 4608 cross-sections (reference: 384 x 12 = 4608)
    value_epochs: int = 200  # num_epoch
    value_lr: float = 1e-4  # lr (Adam, beta_1 = beta_2 = 0.99)
    value_batch: int = 128  # batch_size (cross-sections)
    value_valid: int = 512  # valid_size (monitoring only)

    # --- policy (reference: policy_config) ---
    num_steps: int = 10000  # num_step
    freq_update_v: int = 2000  # freq_update_v
    policy_lr: float = 4e-4  # lr_beg = lr_end (Adam, beta_1 = beta_2 = 0.99)
    policy_batch: int = 32  # reference: 384 sampled households; here each cross-section is exact over households
    policy_paths: int = 50  # economies per policy data set, re-simulated after each pass (epoch_resample 0)
    cross_sections_per_econ: int = 32  # per economy, at distinct random periods (reference: every 4th of 200)
    #                                    -> 50 x 32 / 32 = 50 steps per pass, as the reference (19200 / 384)
    window: int = 29  # t_unroll: the reference uses 149 periods of utility then beta^149 V with beta = 0.99
    #                   (0.99^149 = 0.224); 0.95^29 = 0.226 gives the continuation the same weight

    # --- evaluation (as GPDP: fixed economies and key) ---
    eval_every: int = 1000  # steps
    eval_envs: int = 32
    eval_seed: int = 12345
