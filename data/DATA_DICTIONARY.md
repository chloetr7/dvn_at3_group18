# DATA DICTIONARY

## Project scope

**Assignment 3 — Topic 1: Compute and algorithms**

This repository uses two Epoch AI source datasets and three processed analytical datasets.

```text
raw_ai_models.csv
    -> prepared_data.py
    -> ai_models.csv

raw_benchmarks.csv
    -> prepared_data.py
    -> benchmarks.csv
    -> filter benchmark_name == "GPQA diamond"
    -> gpqa.csv
```

**Important analytical rule:** benchmark scores are measured source values only. Missing benchmark scores, training compute, or parameter counts are **not imputed or estimated**. `flop_per_param` is treated as a **training-intensity proxy**, not as a direct measure of algorithmic efficiency.

---

# 1. Source datasets

## 1A. `data/raw_ai_models.csv`

**Dataset:** Epoch AI — Notable AI Models  
**Source:** Epoch AI  
**Source URL:** https://epoch.ai/data/ai-models  
**Licence:** Creative Commons Attribution (CC BY); attribution required.  
**Purpose:** Source metadata for model scale, training compute, model size, training data, estimated training cost, domain, organisation, and accessibility.

**Primary key used:** `(Model, Publication date)`  
**Downloaded source coverage:** 1950-07-02 to 2026-09-30  
**Latest source observation:** 2026-09-30

### Source columns used downstream

| Source column | Type | Unit | Use |
|---|---|---|---|
| `Model` | String | Identifier | Model name |
| `Organization` | String | Categorical | Lead organisation |
| `Publication date` | Date | `YYYY-MM-DD` | Release/publication date |
| `Domain` | String | Categorical | One or more model domains |
| `Parameters` | Numeric | Parameters | Trainable parameter count |
| `Training compute (FLOP)` | Numeric | FLOP | Pre-training compute |
| `Training dataset size (total)` | Numeric | Source-native datapoints / tokens / examples | Training-data size where available |
| `Training compute cost (2023 USD)` | Numeric | 2023 USD | Estimated training compute cost |
| `Model accessibility` | String | Categorical | Raw model-access category |
| `Base model` | String | Identifier | Base model where applicable |

### Transform into `ai_models.csv`

1. Select the source columns above.
2. Parse `Publication date` and derive `release_year`.
3. Retain releases from 2018 through 2026 inclusive.
4. Coerce quantitative fields to numeric.
5. Retain rows with positive, non-missing `Training compute (FLOP)` and `Parameters`.
6. Derive base-10 log measures of compute and parameters.
7. Convert parameter counts to billions.
8. Compute `flop_per_param = training_compute_flop / parameters` and its base-10 logarithm.
9. Derive `licence_category` from the raw accessibility field while retaining the original field.
10. Extract the first listed domain as `primary_domain`.
11. Deduplicate on `(model_name, publication_date)`.

---

## 1B. `data/raw_benchmarks.csv`

**Dataset:** Epoch AI — Capabilities & Benchmarking export  
**Source:** Epoch AI  
**Source URL:** https://epoch.ai/benchmarks/use-this-data  
**Licence:** Epoch AI data is available under Creative Commons Attribution. Benchmark questions/answers and externally sourced benchmark records may retain the original creators' licences.  
**Purpose:** Source of measured benchmark performance and model-version metadata for capability analysis.

**Rows in source sample:** 1,765  
**Columns in source sample:** 65  
**Primary identifiers used:** `id_runs`, `id_model_version`  
**Measured score field used:** `Best score (across scorers)`

### Source columns used downstream

| Source column | Type | Unit | Use |
|---|---|---|---|
| `id_runs` | String | Identifier | Unique benchmark-run ID |
| `id_model_version` | String | Identifier | Model-version ID |
| `Model` | String | Identifier | Model display name |
| `original_task_name` | String | Categorical | Human-readable benchmark name |
| `task` | String | Categorical | Fallback task identifier |
| `Best score (across scorers)` | Numeric | Proportion `0–1` | Primary measured benchmark score |
| `best_score` | Numeric | Proportion `0–1` | Measured fallback score |
| `Version release date` | Date | `YYYY-MM-DD` | Preferred model-version date |
| `Publication date` | Date | `YYYY-MM-DD` | Fallback publication date |
| `Organization` | String | Categorical | Lead organisation |
| `Domain` | String | Categorical | Model domain(s) |
| `Parameters` | Numeric | Parameters | Parameter count where disclosed |
| `Training compute (FLOP)` | Numeric | FLOP | Training compute where disclosed |
| `Training compute cost (2023 USD)` | Numeric | 2023 USD | Training compute cost where available |
| `Model accessibility` | String | Categorical | Raw access category |
| `Country (of organization)` | String | Categorical | Organisation country |
| `Reference` | String | Text / citation | Source reference |
| `Link_model` | String | URL | Model/source link |

### Transform into `benchmarks.csv`

1. Preserve run and model-version identifiers.
2. Use `original_task_name` as `benchmark_name`; fall back to `task` only when needed.
3. Parse `Best score (across scorers)` as the measured `benchmark_score`; use `best_score` only as a measured-source fallback.
4. Convert scores to percentage points with `benchmark_score_pct = benchmark_score * 100`.
5. Do **not** estimate or fill missing benchmark scores.
6. Prefer `Version release date`; use `Publication date` as a fallback.
7. Carry across model metadata embedded in the benchmark export.
8. Derive log-scale compute and parameter variables only where positive raw values exist.
9. Derive `flop_per_param` only where both positive compute and parameter counts exist.
10. Create `has_training_compute` and `has_parameters`.
11. Derive `licence_category` and `primary_domain`.
12. Remove rows missing required run/model/task/score identifiers.
13. Validate measured scores lie in the `0–1` interval.
14. Deduplicate on `(model_version_id, benchmark_name)`.

---

# 2. Processed datasets

## 2A. `data/ai_models.csv`

**Purpose:** Main model-scaling dataset for compute, parameter scale, training intensity, accessibility, domain, and change-over-time analyses.

**Source:** Derived from `raw_ai_models.csv` via `prepared_data.py`  
**Rows:** 351  
**Columns:** 18  
**Primary key:** `(model_name, publication_date)`  
**Duplicate primary-key rows:** 0  
**Time coverage:** 2018-02-01 to 2026-09-09  
**Latest observation:** 2026-09-09  
**Core completeness:** all 351 rows have positive `training_compute_flop` and `parameters`.

### Columns and units

| Column | Type | Unit | Description |
|---|---|---|---|
| `model_name` | String | Identifier | Model name |
| `organization` | String | Categorical | Lead organisation |
| `publication_date` | Date | `YYYY-MM-DD` | Publication/release date |
| `domain` | String | Categorical | Raw comma-separated domain field |
| `parameters` | Float | Parameters | Trainable parameter count |
| `training_compute_flop` | Float | FLOP | Pre-training compute |
| `dataset_size_datapoints` | Float | Source-native datapoints / tokens / examples | Training-data size where available; 62 missing |
| `training_cost_usd` | Float | 2023 USD | Estimated training compute cost; 211 missing |
| `accessibility` | String | Categorical | Raw model-access category; 1 missing |
| `base_model` | String | Identifier | Base model where available |
| `release_year` | Integer | Calendar year | Year derived from `publication_date` |
| `log10_compute_flop` | Float | `log10(FLOP)` | Base-10 logarithm of training compute |
| `log10_parameters` | Float | `log10(parameters)` | Base-10 logarithm of parameter count |
| `parameters_billions` | Float | Billions of parameters | Parameter count divided by `1e9` |
| `flop_per_param` | Float | FLOP / parameter | Training compute per parameter; training-intensity proxy |
| `log10_flop_per_param` | Float | `log10(FLOP/parameter)` | Log training intensity |
| `licence_category` | String | Categorical | Project-derived access grouping: `Open-Weight` or `Closed / API-Only`; not the dataset's legal licence |
| `primary_domain` | String | Categorical | First listed domain |

### Validation

- 351 rows, 18 columns.
- No duplicate primary keys.
- Release years: 2018–2026.
- Training compute range: `9.828e14` to `5.000e26` FLOP.
- Parameter range: `56,804` to `3.0e12`.
- Access grouping: 207 `Open-Weight`, 144 `Closed / API-Only`.

---

## 2B. `data/benchmarks.csv`

**Purpose:** Clean table of measured benchmark observations across all benchmark tasks in the downloaded Epoch AI Capabilities & Benchmarking source.

**Source:** Derived from `raw_benchmarks.csv` via `prepared_data.py`  
**Rows:** 1,749  
**Columns:** 27  
**Primary key:** `benchmark_run_id`  
**Secondary uniqueness constraint:** `(model_version_id, benchmark_name)`  
**Duplicate benchmark-run IDs:** 0  
**Duplicate `(model_version_id, benchmark_name)` pairs:** 0  
**Time coverage:** 2023-03-14 to 2026-10-15  
**Latest observation:** 2026-10-15  
**Distinct benchmark tasks:** 24  
**Distinct model versions:** 411  
**Measured-score policy:** 1,749/1,749 retained rows contain measured benchmark scores; no synthetic scores are present.

### Columns and units

| Column | Type | Unit | Description |
|---|---|---|---|
| `benchmark_run_id` | String | Identifier | Unique benchmark-run ID |
| `model_version_id` | String | Identifier | Model-version ID |
| `model_name` | String | Identifier | Model display name |
| `benchmark_name` | String | Categorical | Benchmark/task name |
| `benchmark_score` | Float | Proportion `0–1` | Measured benchmark score |
| `benchmark_score_pct` | Float | Percentage points `0–100` | `benchmark_score * 100` |
| `benchmark_score_is_measured` | Boolean | Boolean | `True` for all retained rows |
| `publication_date` | Date | `YYYY-MM-DD` | Model-version release/publication date |
| `release_year` | Integer | Calendar year | Year derived from `publication_date` |
| `organization` | String | Categorical | Lead organisation |
| `domain` | String | Categorical | Raw domain field |
| `parameters` | Float | Parameters | Model parameter count where disclosed |
| `training_compute_flop` | Float | FLOP | Training compute where disclosed |
| `training_cost_usd` | Float | 2023 USD | Training compute cost where available |
| `accessibility` | String | Categorical | Raw model-access category |
| `country` | String | Categorical | Country of organisation |
| `reference` | String | Text / citation | Source reference |
| `model_link` | String | URL | Model/source link |
| `log10_compute_flop` | Float | `log10(FLOP)` | Derived where positive compute exists |
| `log10_parameters` | Float | `log10(parameters)` | Derived where positive parameter count exists |
| `parameters_billions` | Float | Billions of parameters | Parameter count divided by `1e9` |
| `flop_per_param` | Float | FLOP / parameter | Derived where both raw inputs exist |
| `log10_flop_per_param` | Float | `log10(FLOP/parameter)` | Logarithm of `flop_per_param` |
| `has_training_compute` | Boolean | Boolean | Whether positive training compute is available |
| `has_parameters` | Boolean | Boolean | Whether positive parameter count is available |
| `licence_category` | String | Categorical | Project-derived model-access grouping |
| `primary_domain` | String | Categorical | First listed domain, or `Unspecified` |

### Missingness relevant to analysis

- Training compute available: **369 / 1,749** rows.
- Parameters available: **507 / 1,749** rows.
- Both compute and parameters available: **271 / 1,749** rows.
- Missing values remain `NaN`; they are not filled or estimated.

---

## 2C. `data/gpqa.csv`

**Purpose:** Primary capability-analysis subset for capability-versus-compute visualisations.

**Source:** Exact filtered subset of `benchmarks.csv` where `benchmark_name == "GPQA diamond"`  
**Rows:** 316  
**Columns:** 27  
**Primary key:** `benchmark_run_id`  
**Secondary uniqueness constraint:** `(model_version_id, benchmark_name)`  
**Duplicate keys:** 0  
**Time coverage:** 2023-03-14 to 2026-10-15  
**Latest observation:** 2026-10-15  
**Benchmark:** GPQA Diamond only  
**Measured score range:** `0.1322601–0.9640152` = `13.23%–96.40%`  
**Measured-score policy:** 316/316 rows are measured observations.

`gpqa.csv` uses the same 27-column schema and units as `benchmarks.csv`.

### Availability for planned analyses

- Total GPQA observations: **316**
- With training compute: **93**
- With parameters: **134**
- With both training compute and parameters: **79**

For capability versus compute:

```python
gpqa_compute = gpqa[gpqa["has_training_compute"]]
```

For an analysis requiring both compute and model size:

```python
gpqa_compute_params = gpqa[
    gpqa["has_training_compute"] & gpqa["has_parameters"]
]
```

---

# 3. Analytical interpretation rules

1. **Measured capability only.** Benchmark scores come from measured source observations; no score is generated from compute, year, parameters, or model name.
2. **Missing is not zero.** Missing benchmark, compute, or parameter values represent unavailable data.
3. **`flop_per_param` is not direct algorithmic efficiency.** It is a training-intensity / compute-allocation proxy.
4. **Avoid causal wording.** The data can show associations between compute, model scale, release time, and measured benchmark performance; it does not establish causality.
5. **Do not pool unlike benchmarks as one capability percentage.** `gpqa.csv` isolates GPQA Diamond for the primary capability analysis.
6. **Show relevant sample sizes.** Capability-versus-compute analysis currently uses the 93 GPQA observations with available training compute.
7. **`licence_category` is an analytical access grouping, not a legal licence.**

---

# 4. Reproducibility

Regenerate all processed files with:

```bash
python3 prepared_data.py
```

Expected outputs:

```text
data/ai_models.csv
data/benchmarks.csv
data/gpqa.csv
```

No synthetic benchmark scores are created. Missing measured values are preserved as `NaN`.
