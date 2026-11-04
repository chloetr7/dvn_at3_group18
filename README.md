# AI Capability: Compute or Efficiency?

DVN36104 Data Visualisation and Narratives — Assignment 3
Group 18

**Deployed app:** [https://dvn-at3-group18.streamlit.app/](https://dvn-at3-group18.streamlit.app/)

**Repository:** [https://github.com/chloetr7/dvn_at3_group18](https://github.com/chloetr7/dvn_at3_group18)

**Assessed version:** [ADD FINAL COMMIT HASH OR RELEASE TAG AFTER THE GROUP FREEZES THE SUBMISSION VERSION]

> The assessed version identifies the exact GitHub state that should be marked.
> Example: commit `a1b2c3d` or release tag `v1.0-final`.

---

## Project Question

**To what extent have recent frontier AI capability gains depended on increasing training compute versus improvements in compute efficiency?**

This project investigates whether recent improvements in frontier AI capability are primarily associated with larger training-compute budgets, or whether newer models also appear to achieve stronger capability at comparable levels of reported training compute.

---

## Target Persona

Our target persona is an **AI Infrastructure Lead at a mid-sized technology company** who needs to decide how much to invest in increasingly compute-intensive models versus more compute-efficient alternatives.

The dashboard is designed to help this user understand:

- how quickly AI training compute has grown;
- how strongly training compute is associated with measured capability;
- whether newer models achieve stronger capability at comparable compute levels; and
- whether models capable of meeting a given performance requirement can be found with lower reported training compute.

---

## Answer in Brief

Across the Epoch AI data used in this project, training compute has increased substantially over time and is positively associated with GPQA Diamond performance.

However, compute does not explain the full variation in measured capability. Models trained within similar compute ranges can achieve substantially different GPQA Diamond scores, and newer models in some comparable-compute ranges achieve higher measured performance than earlier models.

These patterns are consistent with improvements in compute efficiency, but the observational data do not establish that algorithmic progress alone caused the differences. Architecture, training data, post-training methods, model selection, inference-time techniques and missing compute disclosures may also contribute.

---

## User Stories

### User Story 1 — Compute vs Capability

**As an AI Infrastructure Lead, I want to compare measured AI capability with training compute, so that I can assess whether higher-compute models are consistently associated with stronger performance.**

**Dashboard section:** `US1 · Compute vs capability`

**Analysis:**
[TO COMPLETE BY TAB 2 OWNER]

---

### User Story 2 — Capability at Comparable Compute

**As an AI Infrastructure Lead, I want to compare the capability of models released at different times under similar training-compute levels, so that I can assess whether newer models are achieving better performance without proportionally increasing compute.**

**Dashboard section:** `US2 · Capability at comparable compute`

#### Analytical approach

This section uses measured **GPQA Diamond** benchmark observations from Epoch AI.

The preparation process:

1. Starts from the Epoch AI benchmark export.
2. Filters observations to **GPQA Diamond**.
3. Standardises benchmark scores as percentages.
4. Retains observations with reported positive pre-training compute.
5. Excludes missing training-compute observations rather than estimating or imputing them.
6. Groups observations into broad one-order-of-magnitude compute bands.
7. Compares yearly GPQA distributions only where at least three measured observations are available for a release year.

The default view uses the **10²⁴–10²⁵ FLOP** training-compute range.

#### Current finding

Within the default 10²⁴–10²⁵ FLOP range:

- **45** measured observations are included;
- **3** release years contain enough observations to be displayed;
- overall median GPQA Diamond performance is **65.3%**;
- yearly median GPQA increases from **40.7% in 2024** to **89.7% in 2026**, an increase of approximately **49.1 percentage points**.

This result is consistent with capability improvement at broadly comparable reported training-compute levels.

It should **not** be interpreted as causal evidence that algorithmic progress produced the improvement. Differences may also reflect architecture, training data, post-training methods, reasoning techniques and model selection.

#### Visual design

The primary visual is a **line chart of yearly median GPQA Diamond performance**, with interquartile-range error bars.

The analytical task is a comparison of capability over time while constraining training compute to a comparable range.

A scatter plot of all individual observations was considered as an alternative. It was not selected as the primary visual because the main task is to communicate the year-to-year pattern rather than individual-model variation. With many observations, a scatter plot would make the temporal comparison less direct.

The median provides a clearer summary of the yearly pattern, while the interquartile range preserves information about within-year variation.

A release year is shown only when at least **three measured observations** are available.

#### Interaction

Users can select one of the available training-compute bands:

- 10²³–10²⁴ FLOP
- 10²⁴–10²⁵ FLOP
- 10²⁵–10²⁶ FLOP

The KPIs, yearly summary, chart and verification output update to match the selected range.

#### Verification and traceability

A displayed yearly median can be traced through the following data lineage:

`Epoch AI benchmark export`
→ `raw_benchmarks.csv`
→ `prepared_data.py`
→ `benchmarks.csv`
→ GPQA Diamond filter
→ `gpqa.csv`
→ selected compute-band filter
→ yearly median

For example, in the default compute range, the displayed **2026 median GPQA score is 89.74%**, based on **10 measured observations**.

The application independently:

- recalculates the median from the underlying observations;
- verifies that it matches the displayed median;
- verifies that every included observation falls inside the selected compute band; and
- verifies that the displayed observation count matches the underlying rows.

The individual observations used in the calculation can also be inspected directly in the application.

#### Limitations

This analysis controls compute only within a broad order-of-magnitude range, rather than comparing models with exactly equal training FLOP.

It also cannot establish a causal estimate of algorithmic efficiency. Differences between models may reflect architecture, training data, post-training methods, model selection and other factors.

Observations without published training compute are excluded, which may introduce selection bias.

No benchmark scores or compute values are imputed.

---

### User Story 3 — Resource-Efficient Alternatives

**As an AI Infrastructure Lead, I want to identify models that achieve strong measured capability with comparatively lower training compute, so that I can evaluate more resource-efficient alternatives instead of selecting models based on scale alone.**

This user story is explored through two complementary views:

#### User Story 3A — Compute required to reach a capability level

[TO COMPLETE BY TAB 4 OWNER]

#### User Story 3B — Resource-efficient model choices

##### Objective

This dashboard supports an AI Infrastructure Lead in identifying models that achieve strong measured capability with comparatively lower training compute.

##### Data and Methodology

The analysis uses the Epoch AI GPQA Diamond benchmark dataset. Among 316 observations, 93 have reported positive training compute and are eligible for comparison. Missing compute values are excluded rather than imputed.

##### Visualisation Design

A ranked dot plot with a logarithmic horizontal axis is used to compare the training compute of qualifying models.

A standard linear bar chart was considered but rejected because training compute varies by several orders of magnitude, making smaller values difficult to compare.

The dashboard also identifies Pareto-efficient candidates based on benchmark performance and training compute.

##### Interactions

Users can adjust the minimum GPQA Diamond score, release-year range, organisation and number of models displayed. The ranking and summary metrics update based on these selections.

##### Key Finding

With the default minimum GPQA Diamond score of 60%, 43 observations qualify. Among these observations, gpt-oss-20b has the lowest reported training compute of approximately 5.49 × 10²³ FLOP, with a measured GPQA Diamond score of 60.80%.

This finding applies to the selected observations and performance threshold, rather than identifying the universally most efficient AI model.

##### Verification

The dashboard traces displayed values through the original Epoch AI benchmark export, `prepared_data.py`, `benchmarks.csv` and `gpqa.csv`.

For example, the original GPQA Diamond score for gpt-oss-20b is 0.60795455. Multiplying it by 100 and rounding produces the displayed score of 60.80%. Its reported training compute also matches the original record.

##### Limitations

Training compute does not directly measure inference cost, latency or overall algorithmic efficiency. Missing compute disclosures may affect the ranking. The observed relationships do not establish causality.

---

## Dashboard Structure

The application contains five analytical sections:

| Section                                | Purpose                                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Overview · Compute growth              | Establish how rapidly reported AI training compute has increased                                    |
| US1 · Compute vs capability            | Examine the relationship between training compute and measured GPQA capability                      |
| US2 · Capability at comparable compute | Compare capability across release years while constraining compute                                  |
| US3A · Compute to reach a level        | Examine whether a given capability level has later been reached with less compute                   |
| US3B · Efficient model choices         | Identify measured models meeting a selected capability requirement with comparatively lower compute |

---

## Data Sources

### Epoch AI — Notable AI Models

Used primarily for the macro overview of AI training-compute growth.

Source:
[https://epoch.ai/data/ai-models](https://epoch.ai/data/ai-models)

### Epoch AI — Capabilities & Benchmarking

Used for the GPQA Diamond capability analyses.

Source:
[https://epoch.ai/benchmarks/use-this-data](https://epoch.ai/benchmarks/use-this-data)

Licence / attribution: **CC BY**

Full source fields, units, transformations and coverage are documented in:

`data/DATA_DICTIONARY.md`

---

## Data Processing

Raw source files are stored in the `data/` directory.

The processing pipeline is implemented in:

`prepared_data.py`

Key processed files include:

- `data/ai_models.csv`
- `data/benchmarks.csv`
- `data/gpqa.csv`

The processed datasets are generated from the documented source files rather than manually edited dashboard values.

To regenerate the processed datasets:

```bash
python prepared_data.py
```

[ADD ANY REQUIRED COMMAND-LINE ARGUMENTS IF APPLICABLE]

---

## Running the Application Locally

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

---

## Repository Structure

```text
.
├── app.py
├── prepared_data.py
├── requirements.txt
├── README.md
│
├── tabs/
│   ├── tab1.py
│   ├── tab2.py
│   ├── tab3.py
│   ├── tab4.py
│   └── tab5.py
│
├── data/
│   ├── DATA_DICTIONARY.md
│   ├── raw_ai_models.csv
│   ├── raw_benchmarks.csv
│   ├── ai_models.csv
│   ├── benchmarks.csv
│   └── gpqa.csv
│
└── [AI DISCLOSURE / PROVENANCE FILES]
```

Adjust this tree if the final repository structure differs.

---

## Verification and Reproducibility

Displayed analytical results are calculated from the processed datasets rather than manually entered.

Each analytical section includes a verification mechanism that allows selected displayed values to be traced back to their underlying observations and, where applicable, to the raw Epoch AI source export.

The data dictionary documents source provenance, variables, units, time coverage and transformations.

---

## Data and AI Disclosure

Generative AI tools were used as development assistants during the project.

Uses included activities such as debugging code, reviewing Streamlit layout and presentation structure, and refining written explanations.

AI-generated suggestions were not accepted without verification. Team members checked relevant outputs against source data, executed the code, inspected underlying observations and compared displayed calculations with independently recalculated values.

A detailed record of substantive AI use and the verification performed by each team member is provided in:

`[ADD AI PROVENANCE FILE NAME]`

A full Data & AI Disclosure is provided in:

`[ADD AI DISCLOSURE FILE NAME]`

---

## Team Contributions

| Member                    | Main contribution                                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [Member 1]                | [TO COMPLETE]                                                                                                          |
| [Member 2]                | [TO COMPLETE]                                                                                                          |
| Nguyen Thao Nhi Trường    | Project framing and user-story structure; GPQA data preparation; User Story 2 analysis, visualisation and verification |
| [Member 4]                | [TO COMPLETE]                                                                                                          |
| [Member 5, if applicable] | [TO COMPLETE]                                                                                                          |

---

## Data Coverage and Refresh

**Data coverage:** through 09 September 2026
**Last refreshed:** generated when the data-preparation pipeline is run.

The **data coverage date** refers to the latest observation contained in the source data.

The **last refreshed date** should refer to when `prepared_data.py` most recently regenerated the processed datasets. It should not simply use the current time whenever `app.py` runs unless the app itself also refreshes the source data and rebuilds the processed files during that run.

A recommended implementation is for `prepared_data.py` to write the refresh timestamp to a small metadata file, for example:

`data/metadata.json`

and for `app.py` to read and display that timestamp.
