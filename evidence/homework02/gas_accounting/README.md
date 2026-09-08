# Homework 02: balloon-preparation gas records

Prepared teaching dataset for CSE 30124 HW02, following the mansion investigation in HW01. These are synthetic instructional measurements, not real forensic evidence. Quinn's statement rows are carried over from HW01; the filling and weighing records are fictional additions.

Clone the repository as in HW01:

```bash
git clone https://github.com/wtheisen/nd-cse-30124-homeworks.git
```

Students use `historical.csv`, `case_validation.csv`, `case_reserved.csv`, and `scene.csv`. Each prepared row contains `trial_id`, `balloon_volume_liters`, and `mass_loss_g`. IDs are not model features. Raw logs below are retained for provenance; students do not need to join or aggregate them.

Files are under `evidence/homework02/gas_accounting/`. For an existing checkout, use `git pull --ff-only` before starting HW02.

| Files | Role |
|---|---|
| `balloon_specs.csv` | Nominal liters per balloon size |
| `historical_fill_entries.csv`, `historical_weighings.csv` | Fit model weights and preprocessing |
| `case_validation_fill_entries.csv`, `case_validation_weighings.csv` | Select regularization and model candidates |
| `case_reserved_fill_entries.csv`, `case_reserved_weighings.csv` | Evaluate only after the final model is frozen |
| `scene_fill_entries.csv`, `scene_weighings.csv` | Apply the frozen model to the disputed job |
| `hw01_quinn_statements.csv` | Existing narrative evidence |

Each fill row has an entry ID, job/canister keys, time, size, quantity, and status. Only `filled` entries count toward volume. `void` and `unused_return` entries do not consume filling gas. Join size specifications and sum quantity times nominal liters per job/canister. Match before and after weights for the same keys and compute before minus after. Scene records cover 19:55–20:45 for SCENE-C01. Other splits describe separate jobs, not the murder-night timeline.

All splits are public and downloaded together. The assignment requires students to leave reserved outcomes uninspected until model selection is complete, and never use the scene outcome to tune a model. Prepared feature/target tables are supplied; solution notebooks are not included in this directory.
