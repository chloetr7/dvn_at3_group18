# Data & AI Disclosure

DVN36104 Data Visualisation and Narratives — Assignment 3  
Group 18

## Data Disclosure

This project uses public data from **Epoch AI**.

The main data sources are:

- **Epoch AI — Notable AI Models**, used for the macro analysis of training-compute growth.
- **Epoch AI — Capabilities & Benchmarking**, used for GPQA Diamond capability analyses.

The original source exports are retained in the repository alongside processed datasets. Data preparation is performed through `prepared_data.py`.

The project does not intentionally fabricate or impute missing benchmark scores or training-compute values. Observations without the required measured values are excluded where relevant to the analysis.

Detailed information about source URLs, licences, variables, units, time coverage and transformations is documented in:

`data/DATA_DICTIONARY.md`

---

## Generative AI Disclosure

Generative AI tools were used as development assistants during this project.

AI assistance was used for activities including:

- debugging Python and Streamlit code;
- reviewing dashboard layout and information hierarchy;
- discussing visualisation alternatives;
- refining explanatory text and documentation; and
- supporting code review and troubleshooting.

AI-generated suggestions were not treated as authoritative analytical results. Team members remained responsible for the analytical choices, source-data interpretation, final code and claims presented in the application.

Where AI-assisted code or analysis was used, outputs were checked by running the application, examining source or processed data, and comparing displayed results with independently recalculated values.

The table below records substantive AI use by each team member and the verification performed.

---

## AI Provenance Matrix

| Team member | AI tool | Substantive use | Where used | AI contribution | Human verification | Outcome |
|---|---|---|---|---|---|---|
| **Nguyen Thao Nhi Trường** | ChatGPT | Development and review of User Story 2: capability at comparable compute | `tabs/tab3.py`; project framing and documentation | Used ChatGPT to assist with Streamlit debugging, review the layout and information hierarchy of User Story 2, discuss the choice of a yearly-median line chart rather than an individual-model scatter plot, and refine explanatory wording. ChatGPT also assisted with restructuring the tab so the controls, KPIs, main visual, verification and limitations followed a consistent dashboard layout. | The code was run and inspected using the actual processed Epoch AI GPQA data. For the default **10²⁴–10²⁵ FLOP** band, the displayed **2026 median GPQA value of 89.74%** was independently recalculated from the **10 underlying observations**. The verification also tested that every observation fell inside the selected compute band and that the displayed observation count matched the underlying rows. These checks could have failed if the filtering, aggregation or AI-assisted code were incorrect. | The recalculated median matched the displayed value, all observations satisfied the selected compute-band rule, and the displayed count matched the underlying rows. The AI-assisted implementation was retained after these checks. |
| [Member 1] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |
| [Member 2] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |
| [Member 4] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |
| [Member 5, if applicable] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |

---

## Member-Specific AI Use — Nguyen Thao Nhi Trường

### Project framing

ChatGPT was used as a discussion and review tool while structuring the project question, target persona and user-story-based division of the dashboard.

The final project framing and allocation of analytical tasks were decided by the group rather than generated automatically.

### Data and analytical work

For User Story 2, the analysis uses GPQA Diamond observations prepared from the Epoch AI benchmark data.

The analytical rules applied in the final implementation include:

- using GPQA Diamond observations;
- retaining reported training-compute values;
- excluding missing compute rather than imputing it;
- comparing models within broad one-order-of-magnitude compute bands;
- summarising capability using yearly medians;
- showing interquartile ranges; and
- requiring at least three measured observations before displaying a yearly median.

AI assistance was used to review and implement parts of this workflow, but the resulting values were checked against the processed data.

### Visualisation and Streamlit development

ChatGPT was used to assist with debugging and restructuring the Streamlit implementation of User Story 2.

The main visual uses yearly median GPQA Diamond scores with interquartile-range error bars. A scatter plot of individual models was considered as an alternative, but the line chart was retained because the primary task is to compare the year-to-year pattern at comparable compute rather than inspect individual-model variation.

AI assistance was also used to improve the consistency of the page layout with the other dashboard sections.

### Verification

The final application contains an explicit verification section.

For the default **10²⁴–10²⁵ FLOP** compute band:

- the displayed 2026 median GPQA Diamond score is **89.74%**;
- the result is based on **10 measured observations**;
- the application independently recalculates the median from those observations;
- the recalculated result is also **89.74%**;
- every observation is checked against the selected compute-band boundaries; and
- the displayed observation count is checked against the number of underlying rows.

This verification was designed so that an incorrect filter, aggregation or transformation would produce a visible mismatch rather than being accepted without inspection.

---

## Responsibility for Final Outputs

All team members remain responsible for the final submitted work.

Generative AI outputs were treated as suggestions rather than evidence. Final analytical claims were based on the project datasets and were reviewed against the underlying observations before inclusion in the application.
