# Design: `plot_aggy` — aggregated bar/line charts

**Date:** 2026-08-05  
**Status:** Approved for implementation planning (API choices locked in Parameter rules below)  
**Approach:** Extend conjurer’s existing Altair + pandas EDA vis stack (no new plotting library)

## Problem

`plot_histogram` plots frequency of a single series. Common EDA needs are different: **group or bin X, aggregate Y** — e.g. daily cost sum, daily transaction count — as a bar chart (default) or line chart.

This is not a heatmap (`plot_heatmap` is joint *frequency* of X and Y). It is a 1D binning of X with a numeric aggregate of Y.

## Goals

- Public API on `conjurer.eda` that matches the report style of `plot_histogram` / `plot_scatter` (Altair, interactive, conjurer theme).
- Support calendar bucketing (`freq`) and equal-width bins (`num_bins`).
- Support aggregations: `sum`, `count`, `mean`, `min`, `max`, `median`.
- Default mark: bar; optional line.
- Controllable empty-bucket fill.
- Automated unit tests plus a manual notebook (numeric + timestamp), following `test_notebooks/`.

## Non-goals (v1)

- Stacked / grouped bars by a third column (color/facet).
- Cumulative aggregates (running sum).
- New dependencies (Plotly, seaborn, etc.).
- Changing `plot_histogram` semantics.
- Categorical / string / object `column_x` (histogram supports category frequency; `plot_aggy` v1 accepts only numeric or datetime dtypes for X — see X dtype rule).

## Public API

```python
def plot_aggy(
    df,                  # pandas.DataFrame
    column_x,            # str
    column_y,            # str
    agg="sum",           # "sum"|"count"|"mean"|"min"|"max"|"median"
    freq=None,           # calendar Grouper freq ("D","H","W","M",…); if set, ignores num_bins
    num_bins=50,         # equal-width when freq is None
    mark="bar",          # "bar" | "line"
    fill_empty=None,     # None → True if freq else False
    xmin=None,
    xmax=None,
) -> altair.Chart:
    ...
```

Typing in `eda.py` should follow existing helpers (`Union` / `Orderable`), not require Python 3.10+ syntax.

### Examples

```python
# Daily cost
eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D")

# Daily transaction count
eda.plot_aggy(df, "timestamp", "txn_id", agg="count", freq="D")

# Equal-width numeric X, mean Y
eda.plot_aggy(df, "score", "revenue", agg="mean", num_bins=20)

# Same aggregation as line
eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D", mark="line")
```

### Naming

`plot_aggy` — short conjurer-specific name. Behavior is defined by `column_x` / `column_y` / `agg` / `freq`, not by the function name alone.

### X dtype rule

One acceptance rule, used everywhere below:

- **`freq` set:** `column_x` must be datetime-like (`datetime64` / datetimelike). Otherwise `ValueError`.
- **`freq` is None:** `column_x` must be a **numeric dtype** or **datetime dtype**. String, object, and pandas `category` dtypes → `ValueError`.
- **Do not gate with `binning.is_quantitative`.** That helper routes low-cardinality integers to the categorical histogram path. `plot_aggy` always calls `create_bins_quantitative` for accepted equal-width X (including low-cardinality ints).

### Parameter rules

| Condition | Behavior |
|-----------|----------|
| `freq` is set | Calendar buckets via `pandas.Grouper(key=column_x, freq=freq)`. `num_bins` ignored. X dtype rule: datetime-like. |
| `freq` is None | Equal-width bins via `binning.create_bins_quantitative`, then aggregate `column_y` per bin. X dtype rule: numeric or datetime dtype; else `ValueError`. |
| `agg="count"` | Count rows per bucket; `column_y` may be any column (used only as a stand-in / ignored for values). |
| Other aggs | `column_y` must be numeric. |
| `fill_empty is None` | Defaults to `True` when `freq` is set, else `False`. **Why not always-on like histogram:** histogram’s Y *is* count, so empty bins at 0 complete the distribution. Here empty bins as 0 would mislead for `mean`/`min`/`max`/`median`, and equal-width bins over a wide numeric range are often mostly empty when data is clustered — default drop keeps the chart readable. Calendar `freq` defaults to fill because missing periods on a continuous timeline are usually meaningful (quiet days). Callers set `fill_empty=True` on equal-width to mirror histogram’s full bin grid. |
| `fill_empty=True` | Expand full bucket index over `[xmin, xmax]` (or data min/max). `sum`/`count` → `0`; `mean`/`min`/`max`/`median` → NaN for empty buckets. |
| `mark` | `"bar"` → `mark_bar()` with full-bin `x`/`x2` (equal-width edges, or calendar `[start, start+freq)`). `"line"` → layered `mark_line()` + `mark_point()` on the **same** midpoint x (never bar-style `x`/`x2`, which misaligns stroke vs points). |

Invalid `agg` / `mark`, or X/Y failing the dtype rules above → raise clear `ValueError`.

## Architecture

```text
conjurer/eda.py
  plot_aggy(...)  →  aggy.plot_aggy(...)

conjurer/logic/eda/vis/aggregation.py   # table building
  create_aggregation_table(...)

conjurer/logic/eda/vis/aggy.py          # Altair chart
  plot_aggy(...)
```

### Aggregation table (`aggregation.py`)

**Calendar path (`freq` set):**

1. Optionally filter rows to `[xmin, xmax]` on `column_x`.
2. `df.groupby(Grouper(key=column_x, freq=freq))[column_y].agg(...)` (or `size()` for count).
3. If `fill_empty`, reindex to the full period range and fill per agg rules.

**Equal-width path (`freq` is None):**

1. Build bins with `binning.create_bins_quantitative` (respect `xmin`/`xmax` as `minv`/`maxv`) after the X dtype rule passes.
2. Assign each row to a bin; aggregate `column_y` per bin.
3. Output columns: `{column_x}_lb`, `{column_x}_ub`, and a fixed aggregate column name `agg_value` (chart title/axis label carry `agg(column_y)`).
4. If `fill_empty`, keep empty bins in the table with 0/NaN per fill rules; if `fill_empty` is False, drop bins with no rows.

**Reuse:**

- Bin edge / `QuantitativeBin` logic from `binning.py`.
- Chart sizing / interactive / title patterns from `histogram.py`.

**Do not reuse:**

- `create_frequency_table_2d` / heatmap encoding (joint frequency, not Y aggregation).

### Chart (`aggy.py`)

- **Bar + equal-width:** `mark_bar` with `x`/`x2` binned encoding (same pattern as `plot_frequency_numeric`), Y = `agg_value`.
- **Bar + calendar:** `mark_bar` with temporal `x`/`x2` spanning `[period_start, period_start + freq)` so each bar fills the whole bucket (not a thin mark at the period label).
- **Line (either path):** layered `mark_line()` + `mark_point()` sharing one x encoding (so stroke vertices and points coincide). Equal-width and calendar lines use bin/period **midpoints** — never bar-style `x`/`x2` spans for the line mark.
- Title e.g. `{agg}({column_y}) by {column_x}` (+ freq when set).
- Tooltip: bar → bucket bounds (`lb`/`ub` or period start/end) + `agg_value`; line → those bounds **and midpoint** + `agg_value` (both calendar and equal-width).
- Height/width aligned with histogram (~200×800).

## Error handling

- Empty frame / all-null X after filters → `ValueError` with a short message.
- `min == max` on equal-width → let `binning.BinCreationError` propagate (same as histogram).
- Unknown `agg` / `mark`, or X/Y failing the **X dtype rule** / numeric-Y rules → `ValueError` listing allowed values.

## Testing

### Automated (`tests/eda/`)

Unit-test the **table** (deterministic) and smoke-test the **chart** return type:

| Case | Assert |
|------|--------|
| Calendar `freq="D"`, `agg="sum"` | Daily sums match hand-computed values |
| Calendar `agg="count"` | Daily counts correct |
| Calendar missing day + `fill_empty=True` | Gap day present with `0` for sum/count |
| Calendar missing day + `fill_empty=False` | Gap day absent |
| Equal-width numeric, `agg="mean"` | Per-bin means match |
| `agg` in `{min,max,median}` | Spot-check one calendar or binned case each |
| `mark="line"` | Returns `altair.Chart` (smoke) |
| Invalid `agg` / X failing X dtype rule (e.g. string X, or numeric X with `freq`) | Raises `ValueError` |
| `xmin`/`xmax` | Filters / domain respected |

Prefer asserting on the aggregation DataFrame (exported helper or testing via a non-private function `create_aggregation_table`) rather than decoding Vega specs.

### Manual notebook (`test_notebooks/plot_aggy.ipynb`)

Follow existing `plot_histogram.ipynb` / `plot_scatter.ipynb` style:

1. Dummy **numeric** X/Y → `plot_aggy` with `num_bins`, a few aggs, bar + line.
2. Dummy **timestamp** X + cost/count-like Y → `freq="D"` (and maybe `"H"`), including a series with intentional day gaps to eyeball `fill_empty`.
3. Cells call `.display()` or rely on notebook Altair rendering like the other notebooks.

## Implementation order

1. `aggregation.py` + unit tests for tables.
2. `aggy.py` + `eda.plot_aggy` wiring.
3. Chart smoke tests + notebook.
4. Short docstring on `eda.plot_aggy` matching other public helpers.
