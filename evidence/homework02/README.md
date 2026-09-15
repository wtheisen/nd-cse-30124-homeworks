# HW02: balloon-preparation gas records

These fictional records support the current CSE 30124 Homework 02 assignment.
All four CSV files are used by the published notebook.

| File | Purpose |
|---|---|
| `historical.csv` | 12 earlier balloon-filling jobs; fit model weights and scaling here. |
| `case_validation.csv` | 12 separate experiments; select polynomial degree and regularization here. |
| `mansion.csv` | 13 held-out party-preparation jobs, including `SCENE01`; open only after model selection. |
| `hw01_quinn_statements.csv` | Quinn’s statements carried forward from HW01; loaded by the setup cell. |

The three measurement tables contain `trial_id`, `balloon_volume_liters`, and
`mass_loss_g`. Use balloon volume as the input and mass loss as the target;
`trial_id` is an identifier, not a model feature.

Run the notebook’s setup cell to obtain a fresh copy in Colab. For an existing
local checkout, run `git pull --ff-only`.

Earlier datasets, raw logs, intermediate tables, and the instructor’s data
generator were removed from this student-facing folder. They remain available
in repository history before the cleanup commit.
