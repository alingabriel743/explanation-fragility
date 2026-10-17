# Data

The datasets are not part of the repository. `src/fragility/data.py` downloads them into this folder on first use.

| Dataset | File | Source |
|---|---|---|
| Telco Customer Churn | `Telco-Customer-Churn.csv` | IBM sample data, https://github.com/IBM/telco-customer-churn-on-icp4d |
| German Credit (Statlog) | `german_credit.csv` | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| IBM HR Employee Attrition | `HR-Employee-Attrition.csv` | IBM sample data, https://github.com/IBM/employee-attrition-aif360 |

The German Credit file is the original space-separated `german.data`, saved under the name above. Please consult the original sources for the terms under which each dataset may be used.

To use copies stored elsewhere, set the environment variable `FRAGILITY_DATA` to their folder.
