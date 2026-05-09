"""Gamma GLM example for positive continuous severity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glm_factor_optimizer import GLM, split

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


def make_data(rows: int = 2_000, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=rows)
    equipment_type = rng.choice(["compact", "standard", "heavy"], size=rows, p=[0.30, 0.55, 0.15])
    type_effect = pd.Series(equipment_type).map({"compact": -0.10, "standard": 0.0, "heavy": 0.30}).to_numpy()
    mean = np.exp(7.2 + 0.25 * (age < 25) + 0.15 * (age > 65) + type_effect)
    severity = rng.gamma(shape=2.0, scale=mean / 2.0)
    return pd.DataFrame({"severity": severity, "age": age, "equipment_type": equipment_type})


def main() -> None:
    train_df, validation_df, _ = split(make_data())
    glm = GLM(target="severity", family="gamma", prediction="predicted_severity")

    result = glm.optimize(
        train_df,
        validation_df,
        "age",
        fixed_factors=["equipment_type"],
        trials=20,
        n_prebins=6,
        min_bin_size=50.0,
    )
    train_df = glm.apply(train_df, result.spec)
    validation_df = glm.apply(validation_df, result.spec)

    model = glm.fit(train_df, factors=[result.output, "equipment_type"])
    scored = glm.predict(validation_df, model)
    report = glm.report(scored)

    print("Best age severity spec")
    print(result.spec)
    print()
    print("Validation summary")
    print(report["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
