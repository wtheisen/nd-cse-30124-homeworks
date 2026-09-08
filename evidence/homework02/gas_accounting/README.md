# HW02: balloon-preparation gas records

Synthetic instructional measurements for CSE 30124, continuing the HW01 mansion story. These are fictional additions, not real forensic evidence.

Clone `https://github.com/wtheisen/nd-cse-30124-homeworks.git`. For an existing checkout, run `git pull --ff-only` before starting this revised assignment.

Students use three prepared tables with columns `trial_id`, `balloon_volume_liters`, and `mass_loss_g`. The identifier is not a feature.

| File | Use |
|---|---|
| `historical.csv` | 12 earlier jobs; fit weights and scaling here. |
| `case_validation.csv` | 12 separate validation experiments; choose degree and alpha here. |
| `mansion.csv` | 13 held-out jobs from the fictional party preparations; open only after selection. Includes `SCENE01`, the kitchen/pantry preparation window from 19:55 to 20:45. |

The same smooth ordinary-use relationship generates every split: `4 + .02*x + .002*x**2` grams. Gaussian measurement noise has standard deviation 1.2 g for the older historical records and .35 g for validation and the first twelve mansion jobs. SCENE01 is 54 liters with a fixed +.10 g deviation. The historical volumes deliberately leave a gap.

Run `python generate_data.py` with NumPy and pandas to reproduce the files. Seed 8 was deliberately selected to make regularization clearly helpful among the assignment's fixed candidates. This is a designed lesson, not an unbiased experiment about which algorithm is generally best. Students must still select by validation error; Ridge is not guaranteed to win on other data.

Original raw fill and weighing tables are retained for provenance, with weighings updated to match these prepared measurements. `case_reserved.csv` holds the first twelve mansion rows and `scene.csv` the last; they are components of `mansion.csv`, not extra evaluation stages. Raw logs have schematic timestamps; they do not establish who performed a job. `balloon_specs.csv` and `hw01_quinn_statements.csv` are retained. Students need no joins or aggregation, and all model selection must precede opening mansion outcomes.
