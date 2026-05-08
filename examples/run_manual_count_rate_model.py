"""Small manual example using the simple public API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rate_glm_optimizer import RateGLM, split


def make_data(rows: int = 2_000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.25, 4.0, size=rows)
    temperature = rng.normal(20.0, 7.0, size=rows)
    machine = rng.choice(["standard", "compact", "heavy"], size=rows)
    machine_effect = pd.Series(machine).map(
        {"standard": 0.0, "compact": -0.15, "heavy": 0.35}
    ).to_numpy()
    rate = np.exp(-1.8 + 0.04 * np.maximum(temperature - 20.0, 0.0) + machine_effect)
    defects = rng.poisson(rate * hours)
    return pd.DataFrame(
        {
            "defects": defects,
            "hours": hours,
            "temperature": temperature,
            "machine": machine,
        }
    )


def main() -> None:
    train_df, validation_df, _ = split(make_data())
    glm = RateGLM(target="defects", exposure="hours")

    temp_spec = glm.bins(train_df, "temperature", bins=5)
    train_binned = glm.apply(train_df, temp_spec)
    validation_binned = glm.apply(validation_df, temp_spec)

    model = glm.fit(train_binned, factors=[temp_spec["output"], "machine"])
    scored = glm.predict(validation_binned, model)
    report = glm.report(scored, bins=5)

    print("Temperature bin spec")
    print(temp_spec)
    print()
    print("Validation summary")
    print(report["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
