# Pre-registration A3: an operational monitoring task (written 2026-10-02, before any computation)

## Status at the time of writing
- Development data: main grid, seeds 0-4 (default and per-split tuned hyper-parameters, Gaussian
  multiplicative noise, eps = 0.03). Spearman validity and direct-count results on these data are known.
  A descriptive review-budget check of reversal capture (top 10%) on these data was reported by an
  external reviewer (CIES 45.7%, reversal count 73.7%, count with CIES tie-break 74.7%). No other
  monitoring metric has been computed on any data.
- Evaluation data (untouched): a NEW replication of the main grid on seeds 5-9 (`mainfresh`),
  3 datasets x 5 treatments x 5 models x {default, per-split tuned} = 750 configurations, generated
  after this file was hashed. Hyper-parameters for seeds 5-9 are tuned within each new training split
  with the same nested procedure as for seeds 0-4.

## Task (fixed now)
For each configuration (a deployed model on a dataset), an analyst can review a fixed budget of
B = 10% of the N = 100 explained instances. The analyst wants to review the instances whose explanation
shown to decision-makers will change most often under re-measurement.

Target per instance (observed on the K' = 30 independent neighbours, never used for any score):
    Y(x) = sum_c w_c * rate_c(x),  c in {top1, top3set, sign3, reasons4, action},  w_c = 1/5.
Secondary targets: Y_sign3 = rate_sign3 (direction reversals), Y_reasons4 = rate_reasons4.

## Rankers (all computed from the K = 20 diagnostic neighbours only)
- R1 CIES (bounded, as adopted): score = 1 - CIES
- R2 RBO (p = 0.9): score = 1 - RBO
- R3 Uniform-weight index with bounded aggregation: score = 1 - index
- R4 Count aggregate: score = sum_c w_c * count_c (same weights as the target)
- R5 Count of direction reversals: score = count_sign3
- R6 Count aggregate with CIES tie-break: score = R4 + (1 - CIES)/200 (below the smallest count step 1/100)
- R7 CIES + RBO rank average (the paired diagnostic of the manuscript)
For the secondary targets the matched count (count_sign3, count_reasons4) and the matched count with
CIES tie-break are evaluated in place of R4 and R6.
Random selection captures B = 10% in expectation.

## Metrics (fixed now)
Primary: capture share at budget B. Within a configuration, the B*N instances with the highest scores are
selected, ties at the cut-off allocated proportionally (the expectation under random tie-breaking), and
capture = sum of Y over the selection / sum of Y over all instances. Configurations with sum Y = 0 are
excluded and their number reported. Summary = mean capture over evaluable configurations.
Secondary: (a) capture at B = 5% and 20%; (b) average precision for instances with any change
(Y > 0), horizon = the 30 independent neighbours; (c) the same on the fresh mixed-noise seeds 5-9
(`mixedfresh`), labelled as data already used for Amendment A1.
Uncertainty: 95% intervals from 10,000 bootstrap resamples of seeds within each dataset, paired.

## Hypotheses and decision rules (fixed now)
H1. R4 captures more than R1 (paired difference > 0 with the interval above 0) on the evaluation data.
H2. The CIES tie-break adds value: R6 - R4 > 0 with the interval above 0 on the evaluation data.
    Only if H2 holds may the paper claim operational value of CIES as a complement to counting.
H3. R1 vs R2 is reported in both directions without a directional claim.
All results are reported whatever their direction.
