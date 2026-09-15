# HW05: PCA and k-means segmentation

This bundle lets the proposed Fall 2026 HW05 start independently of a student's HW04 run.

- `recovered_page_1.png` through `recovered_page_4.png`: restored RGB pages from the saved HW04 reference run.
- `mask_page_1.png` through `mask_page_4.png`: reference masks, pixel-identical to the HW04 masks.
- `svm_mask_1.png` through `svm_mask_4.png`: frozen SVM predictions from that HW04 run for comparison.

Use page 1 to fit the unsupervised pipeline, page 2 for validation, and pages 3–4 for final evaluation. Use the notebook's label-free rule to identify the ink cluster; do not use held-out masks to choose cluster labels.

The recovered pages and SVM masks were verified byte-for-byte against the saved HW04 outputs. Reference masks describe printed regions, not character identities.
