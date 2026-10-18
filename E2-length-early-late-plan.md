# Plan: Experiment 2. Trajectory length and early/late attribution bias

(Originally drafted as "Experiment 3"; renamed Experiment 2 by the user. Use the `E2_` prefix on every file.)

## 0. Goal and hypotheses

**Question.** Is trajectory length associated with exact-step success, and with the direction of a miss (early vs late), and does that differ by subset (Hand-Crafted vs Algorithm-Generated), model, prompt (A, B, C2), task type and error type?

**Pre-stated hypotheses (to be tested, not assumed):**
- H1: longer trajectories are associated with lower exact-step success.
- H2: among misses, longer trajectories are associated with a higher chance that the predicted step is downstream (late) of the true step; Hand-Crafted misses are often late.

**Wording rule for every table, figure caption and summary:** write "associated with", never "caused by". Trajectories differ in system, task and error type, so none of this is causal.

## 1. Inputs and dependencies

- **Requires Experiment 1 to be complete:** `E1_predictions.csv` for luna, MiMo and Gemma 4 on prompts A, B and C2, on all 184 trajectories (one row per model, prompt, trajectory; fields: trajectory id, subset, parse_ok, parsed_step, step_correct, etc.). Do not reuse partial results. If any of the 9 model x prompt cells is missing, stop and tell me.
- The trajectories (to get `len(history)`, `mistake_step`, `mistake_agent`, `mistake_reason`, `level`, `question`).
- `prompts/taxonomy.yaml` (the 17-mode file hashed in Experiment 1), needed for the error-type labels.
- **Prompt A is the "original" condition** (the all-at-once prompt as originally published); B and C2 are the modified conditions. Keep this labeling in every table and figure.
- Archived runs (C v1 and Kimi-K3 from earlier) are NOT part of the main analysis. They may appear only in an appendix, labeled exploratory.

## 2. Build the analysis dataset (`E2_analysis_data.csv`)

One row per (trajectory_id, model, prompt). 184 x 3 x 3 = 1,656 rows expected; report the actual count.

| Column | Definition |
|---|---|
| trajectory_id, subset | subset is `AG` (Algorithm-Generated) or `HC` (Hand-Crafted) |
| model, prompt | model in {luna, mimo, gemma4}; prompt in {A, B, C2} |
| length | `len(history)`, the number of trajectory steps. Document how it treats any system/question entry, using the same convention that validated the step labels in Experiment 1 |
| log2_length | `log2(length)`; one unit = a doubling of trajectory length |
| true_step | `int(mistake_step)` (0-based) |
| true_pos_rel | `true_step / (length - 1)` (guard length = 1) |
| parse_ok | from Experiment 1 |
| pred_step | parsed step, empty if parse failure |
| step_success | 1 if exact-step correct, else 0. **A parse failure counts as 0**, same as Experiment 1 |
| signed_error | `pred_step - true_step` (negative = early, positive = late), defined only if parse_ok and the prediction is not exact |
| signed_error_norm | `signed_error / length` |
| miss_direction | `early` if signed_error < 0, `late` if > 0, empty otherwise |
| late | 1 if `late`, 0 if `early`, empty if not a parsed miss |
| out_of_range | 1 if `pred_step >= length` or `pred_step < 0`. Count these as `late` (for >= length) or `early` (for < 0), and report how many there are |
| level | the dataset's `level` field if present |
| task_type | derived, section 3 |
| error_type_mode, error_type_family | derived, section 3 |
| label_flag | 1 for records with a label problem (e.g. HC_025, HC_039, HC_050 label discrepancies, or `mistake_step >= length`); same label source as Experiment 1 |

Report, before modelling: per subset, the min / median / max `length`; the correlation between `length` and `true_pos_rel`; and the number of rows, misses, parse failures and out-of-range predictions per model x prompt. If the length ranges of AG and HC barely overlap, say so in the write-up, because the subset-specific slopes then rest on different parts of the length axis.

## 3. Derived variables: task type and error type (not in the original data)

The original 184 trajectories carry no error-mode labels and no task-category labels beyond `level` and the subset. These must be derived, and the derivation is a measurement step that has to be frozen and disclosed.

### 3a. Error type (a property of the TRUE error, derived from `mistake_reason`)
1. **First, look for existing labels.** The Oct 16 chat ("Trajectory error categorization verification") refers to an error-categorization of the trajectories with Claude, GPT and "agreed" label sources and an 11-category scheme. If those label files still exist and cover all 184, use them as `error_type` (keep their category names), and say they came from that earlier work. Report coverage. If they are missing or incomplete, go to step 2.
2. **Otherwise label `mistake_reason` with the 17-mode taxonomy** (`prompts/taxonomy.yaml`, rendered with the same `render_taxonomy` function as Experiment 1). Use a fixed labeler prompt, temperature 0, one label per trajectory, raw outputs saved. Use two different LLM labelers (neither should be one of the three models being judged, if avoidable) and record agreement (percent agreement and Cohen's kappa). The primary `error_type_mode` is the label where both labelers agree; for disagreements, record both and do not adjudicate silently (ask me). Frozen labeler prompt:

```
You will see the human annotator's explanation of why an agent's step in a multi-agent conversation was the decisive mistake. Classify the mistake into exactly one error mode from the taxonomy below, using only the explanation. If no mode fits, answer OTHER.

Taxonomy:
{taxonomy}

Explanation: {mistake_reason}

Answer with exactly one line: Mode: <code or OTHER>
```

3. **Collapse for analysis.** Map modes to six families (Perception, Reasoning, Planning, Action, Verification, Coordination) as `error_type_family`. Most analyses use the family because many modes will have too few trajectories (the 184 are text, so the Perception modes will be nearly empty). Report the label counts per mode and per family before any modelling.

### 3b. Task type
- Use what the data has: `subset`, and `level` if present for most records.
- Also derive a `task_type` from `question` with a fixed labeler (same procedure: temperature 0, raw outputs saved, two labelers, agreement reported). Pre-specified categories, frozen before looking at any accuracy: (1) web search / fact lookup; (2) file, document or data analysis; (3) math, calculation or logic; (4) coding or computation; (5) image, audio or video; (6) other. Merge categories with fewer than 15 trajectories into "other" and say which were merged.
- Labeler prompt: same structure as above, with the six categories and the question text, answer `Type: <number>`.

**Freeze rule:** once the labels are produced and checked for coverage, save them to `E2_labels.csv` with SHA-256, and do not relabel after seeing any accuracy result. Spot-check about 20 random labels per variable myself and report the result; do not change labels based on the spot-check without asking me.

## 4. Primary models (pre-specified; do not change)

Software: Python `statsmodels` GEE. Data in long format, groups = `trajectory_id` (184 clusters), binomial family with logit link, **exchangeable** working correlation, **robust (sandwich) standard errors** (the default `cov_type='robust'`).

Coding (pre-specified): subset reference = `AG`, model reference = `luna`, prompt reference = `A`. `log2_length` is **centered at its sample mean** (`log2_length_c`), which changes the intercept and main effects but not the slope; state this.

### GEE 1: exact-step success
- Outcome: `step_success` (all rows; parse failures count as 0).
- Formula: `step_success ~ log2_length_c + C(subset) + C(model) + C(prompt) + log2_length_c:C(subset)`

### GEE 2: direction of misses
- Outcome: `late` (1 = late, 0 = early), among rows that are **parsed misses** (parse_ok = 1 and step_success = 0, including out-of-range predictions, which are treated as late or early per the sign). Parse failures have no step and are excluded.
- Same formula and same predictors, same clustering by `trajectory_id`.
- This tests whether longer trajectories are associated with misses landing downstream.

### Reported effects (for each GEE)
- **Odds ratio for one doubling of length, separately for each subset**, from the interaction:
  - AG (reference): `exp(beta_log2_length_c)`
  - HC: `exp(beta_log2_length_c + beta_interaction)`
  - 95% CIs from a linear contrast using the **robust covariance matrix** (build the contrast vector from `res.params.index`; use `t_test`/`wald_test`). Report the raw p-value of each.
- The interaction term itself (difference in slopes between HC and AG): coefficient, OR, 95% CI, p.
- Holm-correct the 4 primary slope p-values (2 outcomes x 2 subsets); report raw and Holm-adjusted values. Everything in sections 5 and 6 is exploratory and is not corrected.

### Convergence rule (critical)
Check `res.converged` and capture every warning (non-convergence, singular matrix, perfect separation). **If either GEE fails to converge or raises a warning that invalidates the fit, do NOT change the model ad hoc** (no dropping predictors, no switching correlation structure, no switching link or family). Save the full error and warning text to `E2_GEE_errors.txt`, stop, and ask me before substituting any other analysis.

## 5. Pre-specified sensitivity analyses (reported next to the primary results; they do not replace them)

1. **Exclude parse failures** from GEE 1 (a parse failure is not a step prediction).
2. **Bias-reduced small-sample covariance** (`cov_type='bias_reduced'`) for both GEEs, because 184 clusters is modest.
3. **Direction is mechanically constrained by where the true step is.** A true step at 0 can only be missed late; a true step at the last position can only be missed early. Because HC errors are often at particular positions, GEE 2 can pick this up as a length effect. Add `true_pos_rel` as a covariate to GEE 2 (and to GEE 1) as a sensitivity model and report how the doubling ORs change. If this sensitivity model fails to converge, report that and stop; do not alter it.
4. **Alternative length measure:** repeat GEE 1 and GEE 2 with `log2(token_length)` of the rendered trajectory (A's rendering) in place of `log2_length`. Label it secondary.
5. **Label sensitivity:** repeat with the three flagged HC records removed (HC_025, HC_039, HC_050, or whichever have `label_flag = 1`).

## 6. Breakdowns the user asked for (descriptive first; model-based second)

The user wants the length effect and the early/late bias **for each model, each prompt, each task type and each error type**, with prompt A as the original and B, C2 as the modified conditions. The pooled GEEs above are the confirmatory analysis. These breakdowns are descriptive and exploratory.

### 6a. Descriptive tables (always produced)
For every stratum below, report: n trajectories, n rows, exact-step accuracy (with Wilson 95% CI), n parsed misses, % late among parsed misses (with Wilson CI), median signed error and median normalized signed error, and the median length. Strata:
- model x prompt (9 cells), also split by subset
- model, prompt, subset, each alone
- task_type, `level` (if available), and subset, each overall and crossed with prompt
- error_type_family (and error_type_mode where n >= 15), overall and crossed with prompt, and crossed with subset
- length terciles, **for display only** (not used in any test): short / middle / long thirds of `length`, to show the pattern; label them "display only"

Cells with fewer than 10 rows are shown greyed with "n < 10, descriptive only". Cells with fewer than 5 are suppressed.

### 6b. Stratified model fits (exploratory)
Fit the same GEE form separately within a stratum **only if** it has at least 30 trajectories and at least 10 events and 10 non-events for the outcome. For each, drop the terms that are constant within the stratum (e.g. no `C(model)` inside a model stratum), and report the doubling OR with 95% CI. Strata: each model (pooling prompts), each prompt (pooling models), each task_type, each error_type_family, and model x prompt if the counts allow. Label every one "exploratory, uncorrected" and do not draw confirmatory conclusions from them. If a stratified fit fails to converge, record it as "not estimable" and move on; do not modify it.

### 6c. Interaction tests (exploratory; added models, the primary models stay unchanged)
Two added GEE 1 models, each with the full primary formula plus one extra interaction: (i) `log2_length_c:C(prompt)`; (ii) `log2_length_c:C(model)`. Report the joint Wald test for the added terms. These answer "does the prompt, or the model, change how much length matters". Repeat for GEE 2. Say they are exploratory.

## 7. Figures

1. **`E2_pred_prob_vs_length.png`:** predicted probability of exact-step success against `log2(length)` (x-axis labeled with the actual lengths, e.g. 4, 8, 16, 32), from the primary GEE 1. Separate lines for HC and AG with 95% bands (delta method on the robust covariance), faceted by prompt (A, B, C2), with models as line styles or small multiples. Overlay the empirical points as a smooth (LOWESS) or quantile-binned display, marked "display only".
2. **`E2_signed_error_centered.png`:** signed error (predicted minus true step) against `log2(length)` for parsed misses, with a horizontal line at 0 and the y-axis centered and symmetric (negative = early, positive = late). Points colored by model, facets by prompt and subset, LOWESS smoother per model. Also produce the version with normalized signed error (`signed_error / length`) as a second panel or file.
3. **Breakdown figures (descriptive):** % late among misses and exact-step accuracy, by task_type and by error_type_family, grouped by prompt (A original vs B vs C2), with Wilson CIs and n shown on each bar.

## 8. Deliverables

- `E2_analysis_data.csv` (section 2) and `E2_labels.csv` (section 3, with hashes).
- `E2_GEE_output.txt` (full summaries, convergence status, warnings) and `E2_GEE_output.csv` (all coefficients, robust SEs, p-values).
- `E2_effects_table.csv` with columns: **Outcome | Subset | Effect | Odds ratio | 95% CI | p-value (raw) | p-value (Holm, primary only) | Interpretation**. Rows: the 4 primary doubling ORs (exact-step success: AG, HC; late-miss: AG, HC), the interaction terms, then the sensitivity rows (clearly marked) and the exploratory stratified rows (clearly marked). The Interpretation column uses the phrase "associated with", e.g. "A doubling of trajectory length is associated with a X% [higher/lower] odds of an exact-step hit".
- `E2_breakdown_by_model.csv`, `E2_breakdown_by_prompt.csv`, `E2_breakdown_by_task_type.csv`, `E2_breakdown_by_error_type.csv` (section 6a), and `E2_exploratory_fits.csv` (6b and 6c).
- The two requested figures plus the breakdown figures.
- `E2_notes.md`: counts, length ranges and overlap by subset, the number of out-of-range predictions, label agreement statistics, any non-converged fit with its error text, every deviation from this plan, and the list of what is confirmatory vs exploratory.

## 9. Order of work and checkpoints

1. Confirm Experiment 1 is complete; load `E1_predictions.csv`; build `E2_analysis_data.csv` without the derived labels. **Show me the section 2 report (counts, length ranges and overlap, out-of-range predictions) before continuing.**
2. Derive and freeze the labels (section 3). **Show me label counts, labeler agreement, and 20 random examples per variable before using them.** If agreement is low (kappa < 0.4) or many categories are tiny, stop and ask.
3. Fit the two primary GEEs. If either fails to converge, stop and ask (section 4).
4. Run the sensitivity analyses (section 5), then the breakdowns (section 6), then figures.
5. Write `E2_notes.md`, with a short plain-language summary: the four primary odds ratios with CIs, then the sensitivity checks that changed them, then the exploratory patterns marked as such.

## 10. Caveats to put in the write-up
- Only 184 trajectories (126 AG + 58 HC) are the clusters; every model and prompt row reuses the same trajectories, so the per-model and per-prompt results are not independent samples.
- Subset is confounded with the agent system that produced the trajectory, so a subset-specific slope is associated with system type as well as length.
- Early/late among misses is conditional on a miss and is mechanically limited by where the true step sits.
- Task type and error type are derived labels (not original annotations), so every result that uses them depends on the labeling.
- Gemma 4, MiMo and luna differ in family and scale, and the three prompts differ in output structure, so model and prompt comparisons here are descriptive.
