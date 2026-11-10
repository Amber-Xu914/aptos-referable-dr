# Role D: draft content for slides 7, 8 and 10

Slide numbers and titles follow Jude's framework (`BIOM9027 Group presentation (framework)` on the Drive).

- All numbers below are from the **official test run** (11 Nov), done once after all models were frozen. Threshold: validation sensitivity at least 85%, applied unchanged to test.
- 3-seed models: mean ± SD of the per-seed metrics, each seed with its own validation threshold.
- Speaker notes are suggestions. Say them in your own words.

---

## Slide 7: Test results, threshold, calibration (1:00)

**On the slide**

- **Threshold:** chosen on validation for at least 85% sensitivity, then applied unchanged to the test set (used once).
  - Why: a missed referable eye (FN) can lead to vision loss before the next screen in 1–2 years. An unnecessary referral (FP) costs one specialist visit.
- **Results table (test set, n = 550, 95% bootstrap CI):**

| Model | AUROC | Grade 1 vs 2–4 AUROC | Sens | Spec | Precision (PPV) | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Baseline (colour logreg) | 0.898 (0.871–0.923) | 0.637 | 0.898 | 0.797 | 0.754 | 0.819 | 0.838 |
| Classical (A_rf_all) | 0.976 (0.965–0.986) | 0.892 | 0.840 | 0.960 | 0.936 | 0.885 | 0.911 |
| CNN (B, EfficientNet-B0 fine-tuned, 3 seeds) | 0.984 ± 0.002 | 0.917 ± 0.012 | 0.828 ± 0.005 | 0.973 ± 0.007 | 0.956 ± 0.012 | 0.887 ± 0.006 | 0.914 ± 0.006 |

  (Seed 1 alone: AUROC 0.982, 95% CI 0.973–0.989; sensitivity 0.822, CI 0.771–0.872.)

- Ready-made image of this table: `slide7_results_table.png`.
- **Statistical comparison (paired bootstrap, 2,000 resamples, test set):**
  - CNN vs Part A: AUROC difference +0.005 (95% CI −0.003 to +0.014), **not significant**. Grade 1 vs 2–4: +0.012 (−0.03 to +0.05), not significant.
  - Both CNN and Part A are significantly better than the baseline: +0.08 overall, +0.25 on grade 1 vs 2–4.
  - **Wording for the slide:** "CNN and the classical pipeline perform similarly; both clearly outperform the colour baseline." Don't write "CNN clearly wins".

- **Figure:** `slide7_roc_test.png` (ROC with operating points) or `slide7_sens_spec_bars.png`.
- **Small inset:** `slide8_calibration.png`. ECE: CNN 0.071 ± 0.017, A_rf_all 0.049 (both slightly under-confident in the middle range).

**Speaker notes (about 1 minute)**

> We chose the threshold on validation data only, so the test set stayed untouched until the end. For screening, a missed referable eye is worse than an unnecessary referral, so we fixed sensitivity at 85% or more and accepted the specificity that came with it. On the test set, the fine-tuned CNN, which won on validation, reached sensitivity 83% and specificity 97%. Note that 83% is a little below the 85% we set on validation: a threshold doesn't transfer perfectly to new images, which is exactly why external validation matters. Overall AUROC looks very high, but part of that comes from grade 0 photos being easy to spot by camera type. So we also report grade 1 versus 2–4, which is the clinically hard decision: 0.92 for the CNN against 0.89 for the classical pipeline and only 0.64 for the colour baseline. A paired bootstrap shows the gap between the CNN and the classical pipeline is not statistically significant, so we'd say they perform similarly; both are clearly better than the baseline. The calibration plot shows whether a score of 0.7 really means about 70% risk. Both models are reasonably calibrated (ECE around 0.05–0.07), and slightly under-confident in the middle range.

---

## Slide 8: Errors, successes and bias (1:00)

**On the slide**

- **Example images (2 × 2):** TP, TN, FP and FN, ideally with a Grad-CAM map from B.
- **Referral rate by original grade** (bar chart, `referral_by_grade.png`):
  - File: `slide8_referral_by_grade.png`; confusion matrix: `slide8_confusion_finetune_s1.png` (TN 315, FP 10, FN 40, TP 185).
  - Errors concentrate at the grade 1/2 boundary. Test, CNN seed 1: all 10 false positives were grade 1 and none were grade 0. 31 of 40 misses were grade 2.
  - Grade 4 is missed more often than grade 3. Test, CNN seed 1: 83% vs 93% referred; 5 grade 4 eyes were missed by all 3 seeds. A_rf_all: 88% vs 87%.
- **Why grade 4 eyes were missed** (Jude's review of 8 missed grade 4 eyes, val):
  - 4 laser-treated eyes
  - 2 "quiet" eyes with few visible lesions
  - 1 with disease at the optic disc
  - 1 poor-quality image
- **Bias and generalisability:**
  - single source (Aravind, India) and mixed cameras
  - a camera shortcut for grade 0
  - noisy labels (30 conflicting duplicate groups). The 18 conflicting test images were all classified correctly, so removing them changes test AUROC only from 0.982 to 0.980.
  - no patient IDs
  - disease-enriched data (40% referable)

**Speaker notes (about 1 minute)**

> Most errors happen where clinicians also disagree, at the line between mild and moderate disease. All false positives on the test set were grade 1 eyes, not healthy ones. The more worrying errors are missed grade 4 eyes. When we looked at them, half were laser-treated: the scars replace the lesions the model looks for. In real screening, these patients are usually already under specialist care. Still, it shows the model has learned "many lesions" rather than "severe disease". For generalisability: all images come from one hospital network, grade 0 photos differ by camera, and the labels themselves are noisy. So these results should not be expected to transfer directly to Australian clinics.

---

## Slide 10: Translation and evidence needed (1:05)

**On the slide**

- **Readiness:** research prototype, tested retrospectively on one public dataset. **Not clinically validated** and not suitable for patient diagnosis.
- **Regulation (TGA):**
  - Software that screens for disease is a medical device and needs ARTG inclusion.
  - As decision support for a clinician, it is likely **Class IIa**. A fully autonomous version would be **Class IIb** (classification rule 4.5).
  - Also needed: a quality management system (ISO 13485), clinical evidence, cybersecurity, and change control for retraining.
- **Reimbursement:**
  - MBS items 12325 / 12326 already fund retinal photography in diabetes (since Nov 2016).
  - There is no separate item for AI grading. A new item would need an MSAC application.
- **Evidence and expertise still needed:**
  - external validation on Australian images and cameras, including Aboriginal Medical Services
  - a prospective study against a stronger reference standard (multi-grader adjudication or OCT)
  - an image-quality gate
  - ophthalmologists, GPs, Aboriginal community-controlled health services, regulatory specialists and health economists
- **Benchmark:** IDx-DR, the first FDA-authorised autonomous DR AI (2018), needed a 900-patient prospective trial in primary care (sensitivity 87%, specificity 91%).
- **Concluding answer:**
  > Technically promising on held-out APTOS images (test AUROC 0.98, sensitivity 83% and specificity 97% at our threshold), and potentially valuable as a triage tool to raise screening coverage in primary care. Before adoption, it needs external and prospective validation in Australian settings, TGA approval and a funding pathway.

**Speaker notes (about 1 minute)**

> So how close is this to clinical use? It's a research prototype. We tested it on one public dataset, looking back, and it is not clinically validated. In Australia it would be a medical device. Because a clinician makes the final call, it would most likely be Class IIa with the TGA. Medicare already pays for retinal photos in people with diabetes, but there's no item for AI grading, so the cost would sit inside the existing visit unless MSAC approves a new item. For comparison, the first autonomous DR system approved in the US needed a 900-patient prospective trial. So, to answer the question: our software is technically promising and could be medically valuable as a triage tool. But it would need validation on Australian patients and cameras, a stronger reference standard, TGA approval and a funding pathway before it could be used in practice.

---

**Sources:** see `D_clinical_translation_notes.md` (TGA classification guidance, MBS 12325/12326 via Optometry Australia, MSAC 1181, Abràmoff 2018, Scheetz 2021).
