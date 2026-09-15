# HW03: witness classification

The current perceptron and logistic-regression assignment uses four tables:

| File | Purpose |
|---|---|
| `historical.csv` | Training statements. |
| `validation.csv` | Choose the follow-up threshold. |
| `mansion_resolved.csv` | Held-out evaluation; open after choosing the threshold. |
| `mansion_leads.csv` | Unlabeled leads to rank; no accuracy can be calculated. |

Inputs are `details_matched` and `details_repeated`; `corroborated` is the 0/1 target in the labeled tables. IDs and narrative text are not model features. These records are synthetic classroom evidence.

Legacy letter images were retired from this folder and remain in Git history. The proposed HW04 and HW05 have their own verified image bundles.
