# E2 notes: trajectory length and early/late attribution bias (adapted to Who&When Pro, Experiment 1 results)

Written for: the thesis author. Wording rule: every length result is an **association**, not a cause.

## What was analysed (and how it differs from E2-length-early-late-plan.md)
The plan assumed Experiment 1 on the original 184 trajectories with prompts A, B, C2 and models luna, MiMo and a third model. Experiment 1 was in fact run on **Who&When Pro multi-agent text traces (2,670)** with **prompts B and C only** (prompt A dropped, plan's C2 renamed C). At the user's instruction ("do the analysis for the done trajectories") E2 was adapted and run on what exists:

| Plan | Used here | Why |
|---|---|---|
| 184 trajectories, AG vs HC | 2,670 Pro traces; subset = framework | the data changed |
| Length ranges | Length varies **only** within magentic-one (1-31 entries) and mathchat (4-30; 825 of 920 have length 4). debate=6, macnet=4, metagpt=3, dylan about 11 are constant | a length slope is not identifiable inside those four |
| Primary GEEs | restricted to **mathchat (MC, reference) vs magentic-one (MA)**, the analogue of AG vs HC | same reason |
| prompts A, B, C2; reference A | **B (reference) and C** | A not run |
| three models (luna, MiMo, a third) | **luna (reference), MiMo (partial)**; the third model was dropped | E1 status |
| clusters = trajectory_id (184) | clusters = `seed_group` = hash(framework, task query), 1,141 groups for 2,670 traces; traces from one task share a seed | traces repeat tasks |
| E1 complete required | **MiMo is partial** (B 1,894, C 1,196 of 2,670; run stopped on Hugging Face credits). Unbalanced rows are used as they are. Sensitivity S5 restricts to the 787 primary traces with all four model-prompt cells | user decision |
| error_type from LLM labelers (kappa, spot checks) | Pro's own mode labels (`mode`, family from its prefix); no LLM labelling, so no agreement statistics | labels already in the data |
| task_type from LLM labelers | task category from the paper (benchmark -> STEM / Coding / Deep Search); benchmark also reported. **task_type is nested in framework** (MC=STEM, MA=Deep Search) so it cannot be separated from subset | no extra labels needed |
| `level` | not available | |
| label_flag (HC_025 etc.) | all 0: no `mistake_step >= length`, no known label issues in Pro; sensitivity 5 of the plan replaced by S5 (balanced cells) and S6 (all six frameworks) | |
| length | number of flattened entries (`n_entries`: turns, with mathchat's first user_proxy turn as entry 0). debate/dylan labels are round-level, so for those frameworks a hit is any turn inside the labelled round and signed error is measured to the nearest boundary of the correct set (used only in S6/descriptive tables) | convention that validated labels in E1 |
| token length (S4) | `log2` of o200k token count of the unlabeled A-style rendering | |

Data: `E2_analysis_data.csv` (8,430 rows = luna 5,340 + MiMo 3,090; primary MC+MA = 5,485 rows from 1,731 traces; SHA-256 starts 4478748311750e30). Sections 3's `E2_labels.csv` was not produced because no labelling was done; the label columns are in the analysis data.

## Section 2 report (primary traces, MC + MA)
- Length (entries): MC min 4 / median 4 / max 30 (825 of 920 at length 4); MA min 1 / median 5 / max 31. Ranges overlap, but MC's slope rests on about 95 traces with length > 4.
- corr(length, true_pos_rel): -0.29 overall (MA -0.38, MC -0.10).
- True step is the **last** entry in 79% of MC and 56% of MA traces, and the first in 0% and 11%. This mechanically limits miss direction (a last-step label can only be missed early).
- Rows / hits / parse failures / out-of-range (pred >= length; none < 0), primary set: luna B 1,731 / 1,500 / 0 / 0; luna C 1,731 / 1,423 / 0 / 62; MiMo B 1,236 / 1,032 / 51 / 7; MiMo C 787 / 641 / 6 / 4. Parsed misses: 231, 308, 153, 140. Out-of-range predictions counted as late (73 total, all in the primary set).

## Primary results (confirmatory; GEE binomial/logit, exchangeable, robust SE, clusters = seed_group; length centred; both fits converged with no warnings)
| Outcome | Effect | OR per doubling [95% CI] | p raw | p Holm (4 slopes) |
|---|---|---|---|---|
| exact-step hit | MC (reference) | 0.43 [0.32, 0.56] | 1.0e-9 | 3.0e-9 |
| exact-step hit | MA | 0.17 [0.13, 0.23] | 7.3e-29 | 2.9e-28 |
| exact-step hit | interaction MA - MC | 0.40 [0.27, 0.61] | 1.3e-5 | n/a |
| late miss | MC (reference) | 1.54 [0.98, 2.40] | 0.060 | 0.119 |
| late miss | MA | 1.86 [0.79, 4.37] | 0.156 | 0.156 |
| late miss | interaction MA - MC | 1.21 [0.47, 3.11] | 0.694 | n/a |

Plain language. **Observed:** a doubling of trajectory length is associated with about 57% lower odds of an exact-step hit in mathchat and about 83% lower odds in magentic-one; the decline is steeper in magentic-one. Doubling is associated with higher odds that a miss lands late (point estimates 1.5 and 1.9) but these intervals include 1 and nothing survives Holm correction. **H1 is supported, H2 is not supported at this sample size** (it is also mechanically confounded, see below).

## Sensitivity (reported next to, not instead of, the primary; `E2_effects_table.csv`)
- S1 exclude parse failures: step ORs 0.38 (MC) and 0.16 (MA), essentially unchanged.
- S2 bias-reduced covariance: same point estimates, slightly wider intervals (late MC 0.93-2.54).
- S3 adding true_pos_rel: the **step slope in MA attenuates (0.17 -> 0.31)** and the MC slope's interval reaches 1 (0.46 [0.21, 1.01]); the late-miss association in MA vanishes (0.98). So part of the length association is explained by where the true step sits (later in longer traces) and the direction result is largely mechanical.
- S4 token length instead of entries: step ORs 0.32 (MC) and 0.34 (MA); the subset difference disappears. Late-miss MC 2.0 [1.1, 3.7] (p=0.026, uncorrected).
- S5 balanced traces (all four cells, 787 traces): step ORs 0.39 and 0.19, same story; late-miss MC 1.74 (p=0.035, uncorrected).
- S6 all six frameworks with framework fixed effects (common slope; length varies only in MA, MC and a little in dylan): step OR 0.22 [0.17, 0.28]; late-miss 1.66 [0.91, 3.05].
- Not run: the plan's label-flag sensitivity (no flagged labels).

## Exploratory (uncorrected; do not draw confirmatory conclusions)
- Interaction tests: length x model for step success, chi2=7.69, p=0.006 (MiMo's slope is shallower than luna's in MC); length x prompt, p=0.23; for late-miss both p>0.6.
- Stratified fits in `E2_exploratory_fits.csv`. For strata containing both subsets the table gives separate `OR_doubling_MC` and `OR_doubling_MA`. Prompt C shows a steeper decline than B in mathchat (OR 0.25 vs 0.90 pooled models, for the MC slope only). Not estimable: luna x B late (iteration limit), MiMo x B late (unstable OR, quasi-separation; not altered), and strata with too few events (math/scibench/simpleqa late, mmlu_pro, Action late).
- Descriptive breakdowns (`E2_breakdown_by_{model,prompt,task_type,error_type}.csv`, for the primary set and for all six frameworks): in MA about 75-94% of parsed misses are **late** (models point downstream of the true step); in MC most misses are **early** (3-26% late), consistent with the true step usually being the last turn.

## Caveats (also from the plan)
- Clusters are tasks (1,141 groups); every model and prompt row reuses the same traces, so model and prompt comparisons are not independent samples. MiMo is a partial, unbalanced subset.
- Subset is confounded with the agent system and task type (MC = STEM, MA = Deep Search); the slope difference is associated with system and task as well as length.
- Direction among misses is conditional on a miss and mechanically constrained by the true step's position.
- MC's slope rests on few traces with length > 4; MC length is mostly 4 because the mathchat transcripts are short.
- Error type is Pro's injected-error label, not an annotation of natural failures; the traces are synthetic.
- Luna and MiMo differ in family and scale; prompts B and C differ in output structure and C has no step headers.

## Confirmatory vs exploratory
Confirmatory: the four primary slope p-values (Holm over 4). Everything else (sensitivity S1-S6, interaction tests, stratified fits, breakdown tables, figures) is exploratory or descriptive.

## Files
`E2_analysis_data.csv`, `E2_GEE_output.txt/.csv`, `E2_GEE_errors.txt` (no errors or warnings in any model-level fit; stratified fit failures are recorded in `E2_exploratory_fits.csv`), `E2_effects_table.csv`, `E2_breakdown_by_*.csv`, `E2_exploratory_fits.csv`, figures `E2_pred_prob_vs_length.png`, `E2_signed_error_centered.png`, `E2_signed_error_norm_centered.png`, `E2_breakdown_{task_type,error_family,error_mode}.png`. Code in `src/` (build.py, gee_lib.py, primary.py, run_all.py, breakdown.py, figures.py). Rerun after MiMo finishes to refresh everything.
