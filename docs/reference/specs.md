# Binning and Grouping Specs

All specs are plain JSON-serializable dictionaries. Specs are learned from
training data and then applied to train, validation, holdout, or future data.

## Numeric Spec

Example:

```json
{
  "type": "numeric",
  "column": "machine_age",
  "output": "machine_age_bin",
  "method": "quantile",
  "edges": [17.999999999, 25.0, 35.0, 50.0, 85.000000001],
  "labels": ["bin_1", "bin_2", "bin_3", "bin_4"]
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `type` | Always `"numeric"`. |
| `column` | Raw input column. |
| `output` | Transformed output factor column. |
| `method` | Source method, such as `"quantile"` or `"optuna_quantile_subset"`. |
| `edges` | Ordered numeric cut edges. |
| `labels` | Bin labels. |
| `prebin_edges` | Optional Optuna prebin edge list. |

Application behavior:

- Values inside edges are mapped to labels.
- Missing or nonnumeric values become `"missing"`.
- The first interval includes the lowest edge.

## Categorical Spec

Example:

```json
{
  "type": "categorical",
  "column": "equipment_type",
  "output": "equipment_type_group",
  "order": ["compact", "standard", "heavy", "specialized"],
  "cutpoints": [2, 3],
  "mapping": {
    "compact": "group_01",
    "standard": "group_01",
    "heavy": "group_02",
    "specialized": "group_03"
  },
  "labels": ["group_01", "group_02", "group_03"],
  "default": "other",
  "missing": "__missing__",
  "stats": []
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `type` | Always `"categorical"`. |
| `column` | Raw input column. |
| `output` | Transformed group column. |
| `order` | Training categories ordered by observed risk or mean target. |
| `cutpoints` | Boundaries over the risk-ordered category list. |
| `mapping` | Category-to-group mapping. |
| `labels` | Group labels. |
| `default` | Value for unseen categories. |
| `missing` | Internal missing key used during grouping. |
| `stats` | Training risk table records used to build the grouping. |

Application behavior:

- Categories are converted to strings.
- Missing categories use the `missing` key.
- Unseen categories become `default`, usually `"other"`.

## Interaction Spec

Interaction specs are created by `GLMStudy.test_interaction(...)`:

```json
{
  "type": "interaction",
  "columns": ["machine_age_bin", "equipment_type_group"],
  "output": "machine_age_bin__x__equipment_type_group",
  "method": "string_cross"
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `type` | Always `"interaction"`. |
| `columns` | Accepted transformed columns to cross. |
| `output` | Interaction output column. |
| `method` | Current method, `"string_cross"`. |

The interaction output is a string cross of the two accepted transformed
columns. Interactions are only added to the model after explicit acceptance.

## Persistence Contract

Specs should remain:

- JSON serializable
- independent of Python objects
- learned from training data only
- directly applicable to validation, holdout, or scoring data
- suitable for logging and audit review
