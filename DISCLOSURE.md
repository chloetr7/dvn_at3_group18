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
| Yuchao Zong | Claude (Anthropic) | Development of the macro overview: training-compute growth | tabs/tab1.py; Macro Overview section of README.md | Claude wrote most of the Streamlit code for Tab 1, including the log-scale scatter plot with trend line, the yearly-median comparison of Open-Weight and Closed / API-Only models, the KPIs, the verification panel and the limitations text. It also helped draft the README section and the Part 1 presentation script. | Ran the tab locally on the real Epoch AI data and checked that the model counts matched DATA_DICTIONARY.md (351 models: 207 Open-Weight, 144 Closed / API-Only). Recalculated the displayed 2026 Open-Weight median (4.32 × 10²⁴ FLOP) from its 17 underlying models, and traced a displayed model back to raw_ai_models.csv to compare training FLOP and parameters. Visually reviewed the charts in the dark theme. These checks could have failed if the filtering, aggregation or lookup were wrong. | Counts, the recalculated median and the raw-record values all matched. The visual review found two real problems in the AI-generated version: model labels and the trend line were invisible in dark mode, and the cumulative growth was shown with false precision (×3,298,754). Both were fixed before committing. |
| [Member 2] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |
| [Member 4] | [AI TOOL] | [SUBSTANTIVE USE] | [FILE / SECTION] | [WHAT AI HELPED WITH] | [SPECIFIC CHECK THAT COULD HAVE FAILED] | [RESULT OF CHECK] |
| **Xingqi Zhang** | ChatGPT by OpenAI | Development and debugging of User Story 3B: resource-efficient model choices | `tabs/tab5.py`; US3B README documentation | ChatGPT assisted with Streamlit code, interactive filters, a ranked dot plot, Pareto-efficient model identification, source verification and debugging. | Ran the dashboard locally using Epoch AI GPQA data. Checked that the raw score of 0.60795455 for gpt-oss-20b converts to 60.80%, and that the raw and processed training compute both equal 5.49e23 FLOP. These checks could have failed if the data lookup or conversion was incorrect. | The score conversion and compute values matched the original benchmark record. The dashboard ran successfully and was committed to GitHub. |


---

## Member-Specific AI Use — Nguyen Thao Nhi Truong

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
## Member-Specific AI Use — Xingqi Zhang (US3B)

### Project contribution

I was responsible for developing User Story 3B, which focuses on identifying AI models that achieve strong measured capability with comparatively lower training compute.

My contribution was implemented in `tabs/tab5.py` and integrated into the group's Streamlit application.

ChatGPT was used to assist with understanding the implementation requirements, structuring the dashboard, developing Python and Streamlit code, and troubleshooting technical issues.

### Data and analytical work

The analysis uses GPQA Diamond benchmark observations from the Epoch AI dataset, prepared through the group's existing data-processing pipeline.

The final implementation follows these analytical rules:

- Use measured GPQA Diamond benchmark scores.
- Retain observations with positive reported training compute.
- Exclude missing compute values rather than imputing them.
- Apply an adjustable minimum capability threshold.
- Rank qualifying observations by reported training compute.
- Identify Pareto-efficient candidates within the selected comparison group.

The processed dataset contains 316 GPQA observations, of which 93 have usable training-compute information.

ChatGPT assisted with implementing the filtering, ranking and comparison logic. The resulting dashboard was tested locally using the actual project data.

### Visualisation and Streamlit development

ChatGPT helped develop the interactive Streamlit interface, including the minimum GPQA score slider, release-year and organisation filters, model ranking visualisation and source verification panel.

The main visualisation is a ranked dot plot using a logarithmic horizontal axis.

A standard linear-axis bar chart was considered as an alternative. However, reported training compute varies by several orders of magnitude, making smaller values difficult to compare on a linear scale.

The dot plot was retained because it provides a clearer comparison of training-compute requirements across qualifying models.

The dashboard also highlights Pareto-efficient candidates to help users examine the trade-off between measured capability and training compute.

### Debugging and testing

ChatGPT was used to help resolve Streamlit compatibility and DataFrame display errors encountered during local development.

After modifying the code, I reran the application and checked that the main visualisations, model selection controls and source verification section displayed correctly.

I also tested the minimum GPQA score filter and confirmed that changing the threshold updated the displayed results.

### Verification and limitations

The final dashboard includes an explicit source verification section.

Under the default minimum GPQA score threshold of 60%:

- 93 observations have usable training-compute information.
- 43 observations meet the selected performance requirement.
- gpt-oss-20b has the lowest reported training compute among the qualifying observations.

For gpt-oss-20b, the original measured GPQA score is 0.60795455.

Multiplying this value by 100 gives 60.795455%, which rounds to the displayed value of 60.80%.

The original and processed training-compute values both match at 5.49 × 10²³ FLOP.

These verification checks could have failed if the benchmark-run lookup, percentage conversion, filtering logic or training-compute transformation were incorrect.

The application displayed matching results for the checked observation.

AI-generated suggestions were evaluated through local code execution and comparison with the source data.

Finally, the analysis does not establish causal algorithmic efficiency, and training compute should not be interpreted as a direct measure of inference cost or real-world operational efficiency.
---
## Member-Specific AI Use — Yuchao Zong (Macro Overview)

### Project contribution

I was responsible for the macro overview tab (`tabs/tab1.py`), which shows how quickly the training compute behind notable AI models has grown since 2018 and compares Open-Weight with Closed / API-Only models. I also wrote the Macro Overview section of the README.

### How AI was used

Claude (Anthropic) was used as a coding and writing assistant. It generated most of the Streamlit code for the tab, following the structure and conventions already used in the other tabs, and later made the layout changes agreed in the group (a written takeaway box and a renamed verification section). Claude also helped draft the README section and my Part 1 presentation script.

The analytical rules in the final tab are:

- use `data/ai_models.csv` only, with no imputed compute or parameter values;
- estimate growth with an ordinary least-squares fit of log₁₀(training FLOP) on release date;
- show yearly medians only where a year and access group has at least three models; and
- present the results as descriptive, not causal.

### Visualisation decision

The main chart is a scatter plot of every model on a logarithmic compute axis with a fitted trend line. A linear axis was rejected because compute spans more than ten orders of magnitude, so the largest recent runs would flatten every earlier model to the floor.

### Verification

- Ran the tab locally on the real data and confirmed the counts against the data dictionary: 351 models, of which 207 are Open-Weight and 144 Closed / API-Only.
- Recalculated the displayed 2026 Open-Weight yearly median (4.32 × 10²⁴ FLOP) from its 17 underlying models; it matched.
- Traced a displayed model back to `data/raw_ai_models.csv` by model name and publication date; training FLOP and parameter count matched exactly.
- Reviewed the rendered charts. This found two problems in the AI-generated code: labels and the trend line were invisible in the dark theme, and the cumulative growth was displayed with false precision (×3,298,754 instead of about 3.3 million times). Both were corrected.

These checks could have failed if the filtering, aggregation, raw-record lookup or display formatting were wrong. I remain responsible for the final code and the claims shown in the tab.

## Responsibility for Final Outputs
