# Pre-registration: CIES variants (written 2026-09-29, before any variant was computed)

## Motivation (from results already observed)
On the main grid, the original CIES predicted decision consequences less well than
RBO (mean Spearman 0.267 vs 0.321). Two structural causes were identified from the
definition and the synthetic benchmark, not from tuning on real data:
1. weights come only from the reference ranking, so a feature that *enters* the top
   ranks after re-measurement receives a small weight;
2. by design, CIES is insensitive to order swaps between near-tied features.

## Variants (fixed now; no parameter will be tuned)
Notation: r_j = rank of |phi_j(x)|, r'_jk = rank of |phi_j(x'_k)| (ordinal, 1 = largest),
h(r) = (1/r)/H_M.

- **V0 CIES** (original): w_j = h(r_j); RI = mean_k sum_j w_j|phi_j - phi'_jk| / sum_j w_j|phi_j|; score = max(0, 1 - RI).
- **V1 CIES-sym** (symmetric weights): w_jk = max(h(r_j), h(r'_jk)) / sum_j max(...);
  RI = mean_k sum_j w_jk|phi_j - phi'_jk| / mean_k sum_j w_jk (|phi_j| + |phi'_jk|)/2; score = max(0, 1 - RI).
- **V2 SRI** (signed-rank index): s_j(phi) = sign(phi_j) h(r_j);
  score = 1 - mean_k sum_j |s_j(phi(x)) - s_j(phi(x'_k))| / 2   (in [0, 1]).
- **V3 CIES\*** = (V1 + V2) / 2   (magnitude + order + direction, equal weights).
- **V4** = (V0 + V2) / 2          (checks whether symmetric weights matter inside V3).

## Evaluation criteria (identical to those already used for all other measures)
- Primary: `mean_decision` = mean over the five decision consequences (top1, top3set,
  sign3, reasons4, action) of the configuration-averaged Spearman correlation between
  instability (diagnostic neighbours) and the consequence rate (independent neighbours).
- Secondary: test-retest reliability; synthetic benchmark AUCs.

## Selection and confirmation rule
1. Development data: main grid, seeds 0-2. Select the variant among V1-V4 with the highest
   `mean_decision`, subject to test-retest reliability >= 0.85.
2. Confirmation data (not used for selection): main grid seeds 3-4; the mixed-noise grid
   (all seeds); the synthetic benchmark.
3. The selected variant is adopted as the revised CIES only if, on BOTH confirmation grids,
   its `mean_decision` exceeds that of V0; on the synthetic benchmark it must keep
   AUC(B vs A) <= 0.60 (no false alarms for marginal-feature instability) and
   AUC(C vs A) >= 0.90.
4. Its comparison with RBO is reported whatever the outcome. All variants are reported.
5. If the rule is not met, the original CIES is retained and the variants are reported as
   tested-and-rejected.

---

# Amendment A1 (written 2026-09-29, before any bounded variant was computed)

## Motivation
On the ORIGINAL mixed-noise grid (all seeds, already inspected), the test-retest reliability of
CIES fell to 0.41 (RBO 0.67). Diagnosis from the definition: RI averages unbounded per-neighbour
changes, so a single neighbour in which a leading categorical field is mis-recorded dominates.
Because this weakness was observed on data designated for confirmation, the amendment is
confirmed only on NEW data (seeds 5-9) that have not been generated yet.

## Bounded variants (fixed now)
Per-neighbour relative change RI_k = D_k / N_k, where for V0: D_k = sum_j w_j|phi_j - phi'_jk|,
N_k = ||phi(x)||_w; for V1: D_k and N_k as in V1 with weights w_jk.
- **V0b** = mean_k max(0, 1 - RI_k)     (bounded original CIES)
- **V1b** = mean_k max(0, 1 - RI_k)     (bounded symmetric CIES)
- **V3b** = (V1b + V2) / 2              (V2 is already bounded per neighbour)

## Confirmation rule for A1
Confirmation data: mixed-noise grid with seeds 5-9 (new splits, instances and perturbations;
default hyper-parameters; all treatments and models). A bounded variant replaces its unbounded
counterpart only if, on these data, it improves BOTH test-retest reliability and mean_decision,
AND on the numeric-noise main grid (seeds 3-4) its mean_decision is not lower by more than 0.01,
AND it satisfies the synthetic criteria of the main pre-registration
(AUC(C vs A) >= 0.90, AUC(B vs A) <= 0.60). The main pre-registration's selection rule for
V1-V4 is unchanged and is applied first.

---

# Outcome of A1 (recorded 2026-09-30 before A2)
On the fresh seeds 5-9 (mixed noise), V0b improved mean_decision (0.146 -> 0.152) and
test-retest (0.396 -> 0.428); on main-grid seeds 3-4 it was not lower (0.263 vs 0.263); it meets
the synthetic criteria (0.967, 0.499). Per the rule, V0b (bounded CIES) replaces V0 as CIES.
V1-V4 selection: V2 (SRI) was best on dev but failed the synthetic criterion -> not adopted.

# Amendment A2 (written 2026-09-30, before any margin bound was computed)

## Idea (derived from the definition, not from data)
Let phi = phi(x), w the harmonic weights, N = ||phi||_w, RI_k = D_R^(k)/N and
CIES = 1 - mean_k min(1, RI_k) (bounded CIES, adopted under A1). For a decision artefact A(phi)
whose change requires some attribution changes of total weighted size at least m_A(x), every
neighbour k with a change has RI_k >= mu_A = m_A/N, so by Markov's inequality
    P(A changes) <= E[min(1,RI)] / min(1, mu_A) = (1 - CIES_inf) / min(1, mu_A).
Plug-in estimate: B_A(x) = min(1, (1 - CIES(x)) / min(1, mu_A(x))), from the K diagnostic neighbours.

## Margins (fixed now)
- top-k set (k = 1, 3), S = top-k by |phi|:  m = min_{l not in S} w_l (|phi_(k)| - |phi_l|)
- direction reversal of a top-3 feature:     m = min_{j in S_3} w_j |phi_j|
- primary risk driver a = argmax phi:          m = min_{l != a} min(w_a, w_l) (phi_a - phi_l)
- adverse-action reasons R (<=4 largest positive phi):
      m = min( min_{j in R, l not in R} min(w_j,w_l)(phi_j - phi_l),  min_{j in R} w_j phi_j,
               [if |R| < 4] min_{l not in R} w_l |phi_l| )
(Each is a valid lower bound on D_R for any neighbour whose artefact changes.)

## Evaluation (fixed now)
Data: main grid (dev seeds 0-2, confirmation seeds 3-4), mixed2 (all seeds), mixedfresh (5-9),
synthetic benchmark. For each consequence c in {top1, top3set, sign3, reasons4, action}:
 (a) coverage: share of instances with observed rate on the K'=30 independent neighbours <= B_c;
 (b) informativeness: share of instances with B_c < 1, and mean(B_c - observed);
 (c) predictive validity: Spearman(B_c, observed rate) per configuration, averaged; mean over the
     five consequences = mean_decision_bound.
Reporting rule: the bound is reported as a theorem regardless. The consequence-specific bound
("certified reason stability" = 1 - B_c) is recommended in the paper as the decision-facing
companion of CIES only if, on ALL confirmation data (main seeds 3-4, mixed2, mixedfresh), its
coverage is >= 0.90 for every consequence AND mean_decision_bound >= mean_decision of CIES.
Its comparison with RBO is reported whatever the outcome.
