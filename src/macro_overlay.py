"""Forward-looking macro overlay (IFRS 9).

Links PD to a macro factor, defines baseline / adverse / upside scenarios,
and returns the probability-weighted ECL. Scenario weights are an expert
judgement and therefore an explicit, challengeable input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ecl_engine import COL_PD_12M, COL_PD_LIFETIME, compute_ecl


@dataclass
class Scenario:
    name: str
    pd_multiplier: float  # how PDs scale vs baseline under this macro path
    weight: float


@dataclass
class ScenarioSet:
    scenarios: list[Scenario] = field(
        default_factory=lambda: [
            Scenario("baseline", 1.0, 0.5),
            Scenario("adverse", 1.6, 0.3),
            Scenario("upside", 0.8, 0.2),
        ]
    )

    def normalised_weights(self) -> dict[str, float]:
        total = sum(s.weight for s in self.scenarios)
        return {s.name: s.weight / total for s in self.scenarios}


def pd_multiplier_from_unemployment(delta_unemployment: float, beta: float = 0.15) -> float:
    """Simple, documented sensitivity: a +1pp rise in unemployment lifts PDs
    by `beta`. Replace with a fitted macro model for a production version."""
    return float(np.exp(beta * delta_unemployment))


def weighted_ecl(df: pd.DataFrame, scenario_set: ScenarioSet | None = None) -> pd.DataFrame:
    """Return per-scenario ECL totals and the probability-weighted total."""
    scenario_set = scenario_set or ScenarioSet()
    weights = scenario_set.normalised_weights()

    rows = []
    weighted_total = 0.0
    for sc in scenario_set.scenarios:
        scen = df.copy()
        scen[COL_PD_12M] = (scen[COL_PD_12M] * sc.pd_multiplier).clip(0, 1)
        scen[COL_PD_LIFETIME] = (scen[COL_PD_LIFETIME] * sc.pd_multiplier).clip(0, 1)
        ecl_total = compute_ecl(scen)["ecl"].sum()
        w = weights[sc.name]
        weighted_total += w * ecl_total
        rows.append({"scenario": sc.name, "weight": w, "ecl": ecl_total})

    rows.append({"scenario": "weighted", "weight": 1.0, "ecl": weighted_total})
    return pd.DataFrame(rows)
