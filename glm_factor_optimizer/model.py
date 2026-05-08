"""GLM fitting for tabular modeling problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def _design_matrix(
    df: pd.DataFrame,
    factors: list[str],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for factor in factors:
        series = df[factor]
        if _is_categorical(series):
            encoded = pd.get_dummies(
                series.astype("string").fillna("missing"),
                prefix=factor,
                drop_first=True,
                dtype=float,
            )
            parts.append(encoded)
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
        parts.append(numeric.fillna(fill_value).to_frame(name=factor).astype(float))

    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    design = sm.add_constant(design, has_constant="add")
    if columns is not None:
        design = design.reindex(columns=columns, fill_value=0.0)
    return design.astype(float)


def _offset(df: pd.DataFrame, exposure: str | None) -> np.ndarray | None:
    if exposure is None:
        return None
    value = pd.to_numeric(df[exposure], errors="coerce")
    if value.isna().any() or (value <= 0).any():
        raise ValueError("Exposure must be positive and non-missing.")
    return np.log(value.to_numpy(dtype=float))


def family_from_name(family: str) -> Any:
    """Return a statsmodels family object for a supported family name.

    Parameters
    ----------
    family:
        Family name. Supported values are ``"poisson"``, ``"gamma"``, and
        ``"gaussian"``.

    Returns
    -------
    object
        Statsmodels family instance.
    """

    normalized = family.lower().strip()
    if normalized == "poisson":
        return sm.families.Poisson()
    if normalized == "gamma":
        return sm.families.Gamma(link=sm.families.links.Log())
    if normalized == "gaussian":
        return sm.families.Gaussian()
    raise ValueError("family must be 'poisson', 'gamma', or 'gaussian'.")


@dataclass(slots=True)
class FittedGLM:
    """Fitted GLM plus the metadata needed for prediction.

    Parameters
    ----------
    model:
        Statsmodels GLM object.
    result:
        Fitted statsmodels results object.
    target:
        Name of the observed outcome column.
    family:
        GLM family name used for fitting.
    factors:
        Factor columns included in the model.
    design_columns:
        Encoded design-matrix columns created during fitting.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column used during fitting.
    prediction:
        Name assigned to prediction output.
    """

    model: Any
    result: Any
    target: str
    family: str
    factors: list[str]
    design_columns: list[str]
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "predicted"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the fitted mean response for new data.

        Parameters
        ----------
        df:
            Data containing the raw factor columns used during fitting.

        Returns
        -------
        pandas.Series
            Predicted mean response indexed like ``df``.
        """

        design = _design_matrix(df, self.factors, columns=self.design_columns)
        predicted = self.result.predict(design, offset=_offset(df, self.exposure))
        return pd.Series(np.asarray(predicted), index=df.index, name=self.prediction)

    def coefficients(self) -> pd.DataFrame:
        """Return fitted coefficients and inference columns.

        Returns
        -------
        pandas.DataFrame
            Coefficient table with term, estimate, standard error, test
            statistic, p-value, and confidence interval columns.
        """

        interval = self.result.conf_int()
        return pd.DataFrame(
            {
                "term": self.result.params.index,
                "coefficient": self.result.params.to_numpy(),
                "std_error": self.result.bse.to_numpy(),
                "z_value": self.result.tvalues.to_numpy(),
                "p_value": self.result.pvalues.to_numpy(),
                "ci_lower": interval[0].to_numpy(),
                "ci_upper": interval[1].to_numpy(),
            }
        )


def fit_glm(
    df: pd.DataFrame,
    target: str,
    factors: list[str],
    *,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    prediction: str = "predicted",
) -> FittedGLM:
    """Fit a GLM with an optional ``log(exposure)`` offset.

    Parameters
    ----------
    df:
        Training data.
    target:
        Observed outcome column.
    factors:
        Factor columns to include in the model.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    exposure:
        Optional positive exposure column used as a log offset.
    weight:
        Optional row-weight column passed as frequency weights.
    prediction:
        Prediction column name stored on the returned model.

    Returns
    -------
    FittedGLM
        Fitted model and prediction metadata.
    """

    design = _design_matrix(df, factors)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = None
    if weight is not None:
        weights = pd.to_numeric(df[weight], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    model = sm.GLM(
        y,
        design,
        family=family_from_name(family),
        offset=_offset(df, exposure),
        freq_weights=weights,
    )
    result = model.fit()
    return FittedGLM(
        model=model,
        result=result,
        target=target,
        family=family.lower().strip(),
        exposure=exposure,
        factors=list(factors),
        design_columns=list(design.columns),
        weight=weight,
        prediction=prediction,
    )


FittedRateGLM = FittedGLM


def fit_rate_glm(
    df: pd.DataFrame,
    target: str,
    exposure: str,
    factors: list[str],
    weight: str | None = None,
    prediction: str = "predicted_count",
) -> FittedGLM:
    """Fit a Poisson GLM for event counts with ``log(exposure)`` as offset.

    Parameters
    ----------
    df:
        Training data.
    target:
        Observed count outcome column.
    exposure:
        Positive exposure column used as a log offset.
    factors:
        Factor columns to include in the model.
    weight:
        Optional row-weight column passed as frequency weights.
    prediction:
        Prediction column name stored on the returned model.

    Returns
    -------
    FittedGLM
        Fitted Poisson model and prediction metadata.
    """

    return fit_glm(
        df,
        target=target,
        exposure=exposure,
        factors=factors,
        family="poisson",
        weight=weight,
        prediction=prediction,
    )
