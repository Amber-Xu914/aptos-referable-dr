# Official test evaluation (Role D), 11 Nov

- **Run:** once, after all models were frozen. Uses `evaluation/evaluate.py` on all 13 prediction files, with the fixed `splits.csv`.
- **Threshold:** the highest threshold that reaches at least 85% sensitivity on validation, applied unchanged to test.
- **3-seed models:** mean ± SD of per-seed metrics, each seed with its own validation threshold.

## Files
| File | What it is |
|---|---|
| `slide7_results_table.csv` | All models, test set, mean ± SD |
| `seed_aggregated_test_metrics.csv` / `per_run_test_metrics.csv` | Full numbers |
| `per1000_at_10pct_prevalence.csv` | Per 1,000 screened at 10% prevalence: unnecessary referrals, PPV, cost at $79.80 per referral (for slide 9b) |
| `slide7_results_table.png` | Slide 7 results table as an image, ready to paste |
| `paired_bootstrap_auroc_difference.csv` | Is CNN significantly better than Part A? (No: +0.005, CI −0.003 to +0.014) |
| `slide7_roc_test.png`, `slide7_sens_spec_bars.png` | Slide 7 figures |
| `slide8_referral_by_grade.png`, `slide8_confusion_finetune_s1.png`, `slide8_calibration.png` | Slide 8 figures |
| `out/<model>/` | Per-model outputs: errors list, triage, calibration, label-conflict breakdown |

## Cross-check against Young's own test metrics (sens90 rule)
- Re-running my code at the 90% rule reproduces Young's test AUROC for all 9 seed runs to 4 decimals.
- TP/FP match exactly for 8 of the 9 runs. The exception is finetune_s1: FP 14 vs 15, because the two scripts pick slightly different thresholds (0.393 vs 0.352) that both give 90% validation sensitivity.

## Main findings
1. The fine-tuned CNN won on validation and has the highest test AUROC, **but its gap to Part A is not statistically significant** (paired bootstrap +0.005, 95% CI −0.003 to +0.014). Say they "perform similarly". Both beat the baseline significantly. Test AUROC is 0.984 ± 0.002 and grade 1 vs 2–4 AUROC is 0.917 ± 0.012, against 0.976 / 0.892 for A_rf_all and 0.898 / 0.637 for the baseline.
2. **Test sensitivity is 0.83, a little under the 85% set on validation.** Most models land at 0.80–0.85. A threshold does not transfer perfectly, so report this honestly.
3. False positives are all grade 1 (none grade 0). Most misses are grade 2. 5 grade 4 eyes were missed by all three CNN seeds: `6cfb7b44ef6f, 7a238a1d3cf3, bfdee9be1f1d, d1a24527a15d, f549294e12e1`. The first 3 were also missed by A_rf_all.
4. All 18 label-conflict test images were classified correctly, so conflicts are not driving the errors. AUROC with them is 0.982; without them it is 0.980 (seed 1).
5. At 10% prevalence the CNN gives about 24 unnecessary referrals per 1,000 people (about $1,900) and a PPV of 0.78. The baseline gives about 183 (about $14,600) and a PPV of 0.33.

6. No data leakage: none of the 123 duplicate-image groups is split across train/val/test.
