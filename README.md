# Explanation fragility in business decision support

Code for the paper *Explanation Fragility in Business Decision Support: A Consequence-Based Evaluation of Stability Measures* by Alin-Gabriel Văduva, Simona-Vasilica Oprea and Adela Bâra (Bucharest University of Economic Studies).

SHAP and LIME explanations can change when the same record is measured again with small recording errors. This repository evaluates whether the measures proposed to quantify this instability anticipate the explanation changes that matter for a decision. Examples of such changes are a different leading reason or a different set of adverse-action reasons. Each measure is computed on one set of type-respecting perturbations of a record and validated against the changes observed on an independent set.

The measures compared are:

- the Critical-feature Index of Explanation Stability (CIES), a rank-weighted index introduced in the paper;
- the stability measures proposed in the literature (sensitivity, local Lipschitz estimates, RIS and ROS, top-k intersection, rank correlations, weighted Kendall's tau, rank-biased overlap);
- direct counts of the explanation changes.

## Repository structure

```
explanation-fragility/
├── src/fragility/          package used by all scripts
│   ├── data.py             download and loading of the three datasets
│   ├── pipeline.py         splits, imbalance treatments, recording-error scenarios, models, SHAP
│   ├── measures.py         CIES and its variants, comparator measures, decision consequences, bound
│   └── stats.py            bootstrap intervals that resample splits within datasets
├── experiments/            scripts that produce the raw results
│   ├── tune.py             hyper-parameter search within every training split
│   ├── run_grid.py         main experiment grids
│   ├── run_synthetic.py    benchmark with known attributions and planted instabilities
│   ├── run_lime.py         LIME sampling randomness versus input perturbation
│   ├── run_smote_factorial.py  why SMOTE appears destabilising under naive noise
│   └── run_scale_check.py  probability versus log-odds attributions
├── analysis/               scripts that turn raw results into tables and figures
│   ├── analyze.py
│   └── analyze_monitoring.py   pre-registered monitoring task
├── preregistration/        pre-registration documents and their SHA-256 hashes
├── data/                   datasets (downloaded on first use, not tracked)
├── run_all.sh              runs the complete study
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

Running the scripts creates `results/`, `tables/` and `figures/` in the repository root. These folders are not tracked.

## Installation

Python 3.14 was used for the study.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps shap==0.50.0
pip install --no-deps -e .
```

On Python 3.14, SHAP 0.50.0 pins pre-release versions of numba and llvmlite. The study used the stable numba 0.67.0 and llvmlite 0.49.0 listed in `requirements.txt`, together with the other dependencies of SHAP. SHAP itself is therefore installed without its dependency pins. Afterwards, `pip check` reports only this pin, which is expected. The procedure was tested in a fresh environment, where it reproduces the stored results of the study.

## Data

The three public datasets are downloaded into `data/` on first use. See `data/README.md` for their sources. To use existing copies, set `FRAGILITY_DATA` to their folder.

## Reproducing the results

```sh
./run_all.sh
```

The script runs the tuning, all experiment grids and the analyses. The grids take several hours on a 14-core machine. They can also be run one at a time, for example `python experiments/run_grid.py main 12`. Every configuration is written to its own file, so an interrupted run resumes where it stopped.

| Grid | Content |
|---|---|
| `main` | 3 datasets, 5 imbalance treatments, 5 models, default and tuned hyper-parameters, seeds 0 to 4 |
| `fresh` | as `main`, seeds 5 to 9 (evaluation data of the monitoring task) |
| `mixed` | default models with mis-recorded discrete fields, seeds 0 to 4 |
| `mixedfresh` | as `mixed`, seeds 5 to 9 |
| `relational` | Telco and HR Attrition with relationships between fields preserved |
| `noise` | four noise models at four noise levels |

## Where the results of the paper come from

| Paper | File produced |
|---|---|
| Table 2 | `tables/T_summary_main.csv` |
| Table 3 | `tables/T_validity_summary_main.csv` |
| Table 4 | `tables/T_monitoring_capture.csv` |
| Figures 2 to 9 | `figures/F_*.png` |
| Table S1 | `tables/T_hyperparameters.csv` |
| Table S2 | `tables/T_prereg_variants.csv` |
| Table S3 | `tables/T_bounds_finite_sample.csv` |
| Table S4 | `tables/T_monitoring_capture.csv` |
| Table S5 | `tables/T_validity_summary_main.csv` (columns `n_*`) |
| Table S6 | `tables/T_paired_by_dataset_main.csv` |
| Table S7 | `tables/T_counts_vs_cies_main.csv` |
| Table S8 | `tables/T_smote_mechanism.csv` |
| Table S9 | `results/synthetic/stable_but_unfaithful.csv` |
| Figure S1 | `figures/F_calibration_main.png` |

Paired differences with confidence intervals are in `tables/T_paired_cis_*.csv`, the effects of the imbalance treatments in `tables/T_imbalance_cis_main.csv`, and the output-scale comparison in `tables/T_scale_check*.csv`.

## Conventions

Ranks are taken by decreasing absolute attribution. Ties are broken in favour of the lower feature index, both for the CIES weights and for the top-k sets of the decision artefacts. No exact ties were found in a sample of 2,500 reference explanations of the study, so the convention does not affect the reported results.

## Pre-registration

The modifications of CIES and the monitoring task were fixed in hashed documents before the corresponding results were computed. The monitoring task is evaluated on seeds 5 to 9, which were generated after its registration. See `preregistration/README.md`.

## License

MIT, see `LICENSE`.
