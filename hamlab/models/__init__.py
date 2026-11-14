"""
Economic models (see base.py for the interface every model provides).

    krusell_smith/  a Krusell-Smith-style economy: saving in capital, Cobb-Douglas prices, an AR(1)
                    productivity shock (MFAX's "endogenous" environment, reproduced exactly)
    huggett/        a Huggett economy with aggregate risk: bonds in zero net supply, the bond price set
                    each period by market clearing (the SRL paper's calibration and tutorial code)
    common/         blocks shared by models (grids, shocks, wealth-and-income transitions)

Each model is a folder with one file per economic block (params, households, firms, markets, shocks)
and model.py, which assembles them: a Model (explicit prices, also an MFAX environment) or a
MarketClearingModel (the price clears a market each period); see base.py.

    from hamlab.models import make
    model = make("krusell_smith")                 # default parameters
    model = make("krusell_smith", sigma=3.0)      # any parameter of KrusellSmithParams overridden

Planned: a one-account HANK model.
"""

from dataclasses import fields

from hamlab.models.base import MarketClearingModel, Model
from hamlab.models.huggett import Huggett, HuggettParams
from hamlab.models.krusell_smith import KrusellSmith, KrusellSmithParams

MODELS = {
    "krusell_smith": (KrusellSmith, KrusellSmithParams),
    "huggett": (Huggett, HuggettParams),
}


def make(name: str = "krusell_smith", **params) -> Model | MarketClearingModel:
    """The model `name`, with any of its parameters overridden by keyword."""
    try:
        cls, params_cls = MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from exc
    valid = {f.name for f in fields(params_cls)}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}; allowed: {sorted(valid)}")
    return cls(params_cls(**params))
