"""Notebook-friendly factor design blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .bins import apply_spec, category_target_order, make_categorical_groups, make_numeric_bins
from .optimize import OptimizationResult
from .validation import by_factor_report

if TYPE_CHECKING:
    from .study import GLMStudy

JsonDict = dict[str, Any]


@dataclass(slots=True)
class FactorBlock:
    """Interactive workbench for one candidate factor.

    Parameters
    ----------
    study:
        Owning :class:`GLMStudy` instance.
    factor:
        Raw factor column managed by this block.
    kind:
        Factor kind, either ``"numeric"`` or ``"categorical"``.
    spec:
        Optional current JSON-serializable binning or grouping spec.
    optimization:
        Optional latest optimization result for this factor.
    target_order_table_:
        Optional category ordering table created by :meth:`target_order`.
    last_comparison:
        Optional latest candidate-vs-current comparison table.
    metadata:
        Free-form notebook metadata about how the current proposal was created.
    """

    study: GLMStudy
    factor: str
    kind: str
    spec: JsonDict | None = None
    optimization: OptimizationResult | None = None
    target_order_table_: pd.DataFrame | None = None
    last_comparison: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> str | None:
        """Return the current proposed output column.

        Returns
        -------
        str or None
            Transformed model column from the current spec, or ``None`` when no
            spec has been proposed.
        """

        if self.spec is None:
            return None
        return str(self.spec["output"])

    def coarse_bins(self, bins: int = 10, method: str = "quantile") -> JsonDict:
        """Create a simple train-derived spec for the factor.

        Parameters
        ----------
        bins:
            Number of numeric bins requested when the factor is numeric.
        method:
            Numeric binning method, currently ``"quantile"`` or ``"uniform"``.

        Returns
        -------
        dict
            JSON-serializable factor spec created from the training sample.
        """

        self.study.require_split()
        if self.kind == "numeric":
            self.spec = make_numeric_bins(
                self.study.train[self.factor],
                bins=bins,
                column=self.factor,
                method=method,
            )
        elif self.kind == "categorical":
            self.spec = make_categorical_groups(
                self.study.train,
                self.factor,
                self.study.target,
                exposure=self.study.exposure,
                weight=self.study.weight,
            )
        else:
            raise ValueError("kind must be 'numeric' or 'categorical'.")
        self.metadata["source"] = "coarse_bins"
        return self.spec

    def target_order(self, max_groups: int | None = None) -> pd.DataFrame:
        """Calculate categorical target order and optionally create a spec.

        Parameters
        ----------
        max_groups:
            Optional maximum number of categorical groups. When supplied, a
            grouped categorical spec is stored on this block.

        Returns
        -------
        pandas.DataFrame
            Category ordering table computed from the training sample.
        """

        self.study.require_split()
        self.target_order_table_ = category_target_order(
            self.study.train,
            self.factor,
            self.study.target,
            exposure=self.study.exposure,
            weight=self.study.weight,
        )
        if max_groups is not None:
            group_count = max(1, min(max_groups, len(self.target_order_table_)))
            cutpoints = _even_cutpoints(len(self.target_order_table_), group_count)
            self.spec = make_categorical_groups(
                self.study.train,
                self.factor,
                self.study.target,
                exposure=self.study.exposure,
                weight=self.study.weight,
                cutpoints=cutpoints,
            )
            self.metadata["source"] = "target_order"
        return self.target_order_table_

    def set_spec(self, spec: JsonDict) -> FactorBlock:
        """Set a manual JSON-serializable spec for this factor.

        Parameters
        ----------
        spec:
            Binning or grouping spec whose ``column`` field matches this raw
            factor.

        Returns
        -------
        FactorBlock
            This block with the manual spec attached.
        """

        if str(spec["column"]) != self.factor:
            raise ValueError("Spec column must match the factor.")
        self.spec = dict(spec)
        self.metadata["source"] = "manual"
        return self

    def optimize(
        self,
        *,
        trials: int = 50,
        max_bins: int = 8,
        n_prebins: int = 10,
        min_bin_size: float | None = None,
        bin_penalty: float = 0.001,
        small_bin_penalty: float = 0.01,
        penalties: Any = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Optimize this factor while keeping accepted factors fixed.

        Parameters
        ----------
        trials:
            Number of Optuna trials.
        max_bins:
            Maximum number of bins or groups allowed.
        n_prebins:
            Number of numeric pre-bins used to define candidate cutpoints.
        min_bin_size:
            Optional minimum bin size override.
        bin_penalty:
            Penalty multiplier for additional bins.
        small_bin_penalty:
            Penalty multiplier for bins smaller than ``min_bin_size``.
        penalties:
            Optional extra penalty callable or named penalty mapping.
        seed:
            Optional Optuna sampler seed.

        Returns
        -------
        OptimizationResult
            Best spec, objective value, and trial history.
        """

        result = self.study.optimize_factor_block(
            self.factor,
            kind=self.kind,
            trials=trials,
            max_bins=max_bins,
            n_prebins=n_prebins,
            min_bin_size=min_bin_size,
            bin_penalty=bin_penalty,
            small_bin_penalty=small_bin_penalty,
            penalties=penalties,
            seed=seed,
        )
        self.optimization = result
        self.spec = result.spec
        self.metadata["source"] = "optuna"
        return result

    def bin_table(self, sample: str = "train") -> pd.DataFrame:
        """Return a grouped summary for the current proposed bins/groups.

        Parameters
        ----------
        sample:
            Split to summarize: ``"train"``, ``"validation"``, or
            ``"holdout"``.

        Returns
        -------
        pandas.DataFrame
            Table with bin label, bin size, observed outcome total, and row
            count.
        """

        self._require_spec()
        df = self.study.sample_frame(sample)
        transformed = apply_spec(df, self.spec)
        return _bin_table(
            transformed,
            output=str(self.spec["output"]),
            target=self.study.target,
            exposure=self.study.exposure,
            weight=self.study.weight,
        )

    def validation_table(self) -> pd.DataFrame:
        """Fit the candidate model and return validation diagnostics.

        Returns
        -------
        pandas.DataFrame
            Validation diagnostics grouped by the candidate factor output.
        """

        self._require_spec()
        comparison, scored = self.study.evaluate_candidate(self.factor, self.spec, return_scored=True)
        self.last_comparison = comparison
        return by_factor_report(
            scored,
            factor=str(self.spec["output"]),
            target=self.study.target,
            prediction=self.study.prediction,
            exposure=self.study.exposure,
            weight=self.study.weight,
        )

    def compare(self) -> pd.DataFrame:
        """Compare the current accepted model against this candidate spec.

        Returns
        -------
        pandas.DataFrame
            One-row table with current and candidate validation deviance.
        """

        self._require_spec()
        comparison = self.study.evaluate_candidate(self.factor, self.spec)
        self.last_comparison = comparison
        return comparison

    def accept(self, comment: str | None = None) -> FactorBlock:
        """Accept this proposed spec into the study.

        Parameters
        ----------
        comment:
            Optional audit note stored in the study history.

        Returns
        -------
        FactorBlock
            This block after acceptance.
        """

        self._require_spec()
        self.study.accept(self, comment=comment)
        return self

    def reject(self, comment: str | None = None) -> FactorBlock:
        """Record that this proposed factor/spec was rejected.

        Parameters
        ----------
        comment:
            Optional audit note stored in the study history.

        Returns
        -------
        FactorBlock
            This block after rejection is recorded.
        """

        self.study.reject(self, comment=comment)
        return self

    def _require_spec(self) -> None:
        if self.spec is None:
            raise ValueError("Create, optimize, or set a spec before using this method.")


def _bin_table(
    df: pd.DataFrame,
    *,
    output: str,
    target: str,
    exposure: str | None,
    weight: str | None,
) -> pd.DataFrame:
    if exposure is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(exposure, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    elif weight is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(weight, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    else:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(target, "size"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    return table.rename(columns={output: "bin"})


def _even_cutpoints(category_count: int, group_count: int) -> list[int]:
    if category_count <= 1 or group_count <= 1:
        return []
    raw = [round(index * category_count / group_count) for index in range(1, group_count)]
    return sorted({int(value) for value in raw if 0 < int(value) < category_count})
