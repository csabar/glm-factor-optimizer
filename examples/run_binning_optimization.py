"""Optimize one numeric factor and one categorical factor manually."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rate_glm_optimizer import RateGLM, split

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


def make_data(rows: int = 2_000, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, size=rows)
    score = rng.normal(size=rows)
    segment = rng.choice(["small", "mid", "large", "enterprise"], size=rows)
    segment_effect = pd.Series(segment).map(
        {"small": -0.25, "mid": 0.0, "large": 0.25, "enterprise": 0.45}
    ).to_numpy()
    events = rng.poisson(np.exp(-0.8 + 0.55 * (score > 0.4) + segment_effect) * exposure)
    return pd.DataFrame({"events": events, "exposure": exposure, "score": score, "segment": segment})


def main() -> None:
    train_df, validation_df, _ = split(make_data(), seed=9)
    glm = RateGLM(target="events", exposure="exposure")

    score_result = glm.optimize(
        train_df,
        validation_df,
        "score",
        fixed=["segment"],
        trials=25,
        n_prebins=8,
        min_bin_size=50.0,
        seed=9,
    )
    segment_result = glm.optimize(
        train_df,
        validation_df,
        "segment",
        kind="categorical",
        trials=20,
        min_bin_size=50.0,
        seed=9,
    )

    print("Best score spec")
    print(score_result.spec)
    print(f"Validation deviance: {score_result.validation_deviance:.6f}")
    print()
    print("Best segment spec")
    print(segment_result.spec)
    print(f"Validation deviance: {segment_result.validation_deviance:.6f}")


if __name__ == "__main__":
    main()
