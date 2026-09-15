# HW04: page recovery and SVM segmentation

The proposed Fall 2026 HW04 uses:

- `scrambled_note_1.png` through `scrambled_note_4.png`: scrambled color page inputs.
- `mask_page_1.png` through `mask_page_4.png`: reference masks for printed regions.

Use page 1 for training, page 2 for validation, and pages 3–4 for held-out evaluation. These files were verified against the original Lab02 source bundle. They are not the different masks from the former HW03.

The assignment exports recovered pages and SVM predictions. A frozen copy is supplied in `evidence/homework05/` so students can begin HW05 independently.

The existing `emnist_balanced_small/` and `segmented_letter_images/` subfolders belong to the older neural-network assignment and remain here for compatibility; they are not inputs to this proposed HW04.
