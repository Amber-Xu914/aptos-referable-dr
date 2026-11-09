# Medical value, workflow and cost-benefit (Part C clinical section)

Scope: patient, clinician and healthcare-system value; workflow integration; cost-benefit. Regulation and reimbursement are led by D. Only the facts needed for the workflow and cost arguments are summarised here (sections 4 and 5), so the two sections stay consistent.

Facts checked against the sources listed at the end (November 2026). Items marked **(secondary source)** were found only in news coverage or search excerpts. Check them before putting them on a slide, or leave them out.

---

## Slide-ready summary

- **The problem is missed screening, not missing technology.** About 1.3 million Australian adults have diabetes [1]. In the National Eye Health Survey, about 1 in 4 non-Indigenous and about 1 in 2 Indigenous people with diabetes were not examined within the recommended interval. Most did not know they needed an exam [3].
- **Intended value:** a referable-DR check at the GP or Aboriginal Medical Service visit the patient is already attending. A negative result means routine rescreening. A positive or ungradable result means referral to an optometrist or ophthalmologist.
- **Value depends on the pathway after the AI.** In an Australian real-world AI screening study, only about a third of referred patients were confirmed to attend [7]. Without a referral pathway, a true positive does not help the patient.
- **Specificity drives cost.** Each false positive is an unnecessary specialist visit. In our validation results, specificity at 85% sensitivity ranges from 0.74 (simple colour baseline) to 0.94 (Part A's random forest). For 1,000 people screened, that is the difference between about 240 and about 55 unnecessary referrals (illustration in section 5).
- **Published evidence for validated systems is positive.** An Australian model found AI screening in primary care cost-saving [9]. Patient acceptability was 94% in the Australian real-world study [7]. **Our student model is not validated and could not be used clinically** (see section 6).

---

## 1. Clinical problem and why it matters

- **Burden.** About 1.3 million Australian adults (6.5% of people aged 18+) had diabetes in 2022–24. Prevalence is higher in outer regional and remote areas (8.0% vs 5.8% in major cities) and in the most disadvantaged group (10.3% vs 2.9%) [1].
- **Diabetic retinopathy (DR).** Among non-Indigenous Australian adults with self-reported diabetes, 28.5% had any DR and 4.5% had vision-threatening DR. About 1 in 10 Indigenous adults with known diabetes had vision-threatening DR. DR was the main cause of vision loss in 9% of non-Indigenous and 19% of Indigenous adults with diabetes [2].
- **Screening gap.** NHMRC guidelines recommend a diabetic eye exam every 2 years for non-Indigenous people and every year for Aboriginal and Torres Strait Islander people. Higher-risk patients need more frequent exams [3, 4]. In the NEHS (2015–16):
  - 77.5% of non-Indigenous and 52.7% of Indigenous participants were examined on time;
  - about 73–74% of those not examined said they did not know an exam was needed [3].
- **Newer data, not directly comparable.** The 2025 Australian Eye and Ear Health Survey reports on-time checks for 82% of First Nations and 71% of non-Indigenous participants with diabetes [4]. The methods differ from NEHS, so the gap is still large but the numbers should not be compared directly.
- **Why the gap matters.** DR is largely asymptomatic until late. Treatment works best before vision is lost, so detection depends on screening, not symptoms.

**Unmet need (one sentence):** many people with diabetes, especially in remote, disadvantaged and Indigenous communities, are not screened on time because eye-care access is limited and the need is not recognised. A fast screen at the point of primary care could close part of that gap.

## 2. Value to patients, clinicians and the healthcare system

| | Potential value | Conditions and risks |
|---|---|---|
| **Patients** | Screening during a visit they already attend; result in minutes instead of a separate optometry or ophthalmology appointment; no travel for the many who are negative. High acceptability in Australian AI screening: 93.7% satisfied, 93.2% would use it again [7]. | A false negative delays treatment of sight-threatening disease (section 3). A false positive causes worry, travel and cost. Small pupils or cataract can make images ungradable (14% in [7]); those patients still need a full exam. |
| **Clinicians (GP, nurse, Aboriginal health worker)** | Turns a skilled grading task into a point-and-shoot photo and an immediate decision. Lets GPs act on the MBS retinal photography items without being expert graders (section 4). | Responsibility stays with the doctor. Staff need training in photography and in explaining results. Staff in [7] reported consent burden, small pupils and gaps in eye-health knowledge. A clear referral pathway was the top factor for medical staff. |
| **Healthcare system** | Specialists see mostly referable cases. Earlier treatment prevents blindness, which is expensive in care, lost work and quality of life. Modelled as cost-saving in Australia [9]. Can reduce inequity where specialists are scarce. | False positives create referral load. Benefit only happens if referred patients attend: about 32% confirmed attendance in [7]. Performance can drop with a different camera or population (section 6). |

## 3. Consequences of errors at the intended threshold

The intended clinical action is: **"If the model is positive (or the image is ungradable), the GP refers the patient to an optometrist or ophthalmologist for a full eye examination."**

| Error | What happens | Harm | How we limit it |
|---|---|---|---|
| **False negative** (referable DR called non-referable) | Patient is reassured and rescreened in 1–2 years | Sight-threatening disease progresses untreated; may reach vision loss before the next screen | Threshold set for high sensitivity (the group uses at least 85% on validation); report sensitivity by grade so missed grade 3–4 cases are visible |
| **False positive** (non-referable called referable) | Unnecessary specialist visit | Anxiety, travel and time off; extra cost to Medicare and patient; uses specialist capacity; frequent false alarms erode trust in the tool | Report specificity at the chosen sensitivity; consider a two-threshold triage band for human grading (D) |
| **Ungradable image** | Must be referred under the MBS rules [5] | Same cost as a false positive | Retake, pupil dilation if allowed, quality-check feedback to the photographer |

**Why sensitivity is prioritised but specificity still matters:** a missed referable case can cost sight, while a false positive costs a visit. In a community population, though, most people screened are negative. At low prevalence, a small drop in specificity produces many false positives (section 5).

## 4. Workflow integration

```
Patient with diabetes attends GP / Aboriginal Medical Service (routine diabetes review)
   │
   ├─ Practice nurse or Aboriginal health worker takes 1 macula-centred colour photo per eye
   │  (non-mydriatic camera; dilation if images are inadequate)        ← trained staff, under GP direction [5]
   │
   ├─ AI software grades the images on site (seconds)
   │
   ├─ Result goes to the GP, who stays responsible and explains it to the patient
   │      ├─ Non-referable → routine rescreen in 2 years (1 year for Aboriginal and Torres Strait Islander
   │      │                   patients or higher risk); reminder via recall system / KeepSight
   │      ├─ Referable     → referral letter to optometrist / ophthalmologist; practice tracks attendance
   │      └─ Ungradable    → referral for a full eye examination
   │
   └─ Results recorded in the patient's GP record (and shared with the eye-care provider)
```

Points to make:
- **Who takes the photo:** MBS items 12325/12326 let trained staff perform the photography under the direction of the medical practitioner, who claims the item and stays responsible [5]. The software therefore fits GP clinics and Aboriginal Medical Services.
- **Pharmacies and optometrists:** they cannot bill these GP items [5]. A pharmacy service would need a different funding model. A pharmacy pilot of a commercial AI DR screener has been announced in Australia, but no outcomes are published **(secondary source)** [12].
- **Who sees the result:** the GP or AMS doctor, not the patient directly. This also matters for regulation: software that informs a health professional is generally classified lower than software that gives the diagnosis directly to the patient (D's section) **(secondary source, check TGA rule 4.5)** [10].
- **The weak point is after the AI.** In [7], 28 patients were referred; only 15 could be contacted and 9 of them attended. Deployment needs:
  - referral tracking;
  - a booked appointment rather than a letter;
  - telehealth or visiting optometry for remote sites.
- **Lessons from other deployments.** In Thailand, Google's system rejected many low-quality images taken in real clinics, frustrating nurses and creating extra follow-ups [13]. Image quality, lighting and connectivity must be planned for. An offline system, as used in [7], avoids the connectivity problem.

## 5. Cost-benefit

**What it costs:**
- Retinal camera and staff time to take photos.
- Software licence or per-screen fee.
- Staff training.
- Referrals generated by positives, false positives and ungradable images.
- Ongoing monitoring of performance.

**What it saves or gains:**
- Specialist and grader time spent on the many negative patients.
- Travel for patients who are negative.
- Above all, blindness prevented by earlier treatment: lost quality of life, care and productivity costs.

**Published evidence (validated commercial systems, not our model):**
- **Australia [9].** Model of 2020–2060 from a healthcare-provider perspective. Universal AI screening in primary care was estimated to:
  - prevent 38,347 cases of blindness in non-Indigenous people with diabetes, gain 172,090 QALYs and save AU$595.8 million (benefit–cost ratio 3.96);
  - prevent 1,211 cases of blindness in Indigenous people with diabetes and save AU$19.2 million (benefit–cost ratio 1.62).
- **Singapore national programme [8].** Annual cost per patient was:
  - US$62 with AI followed by human grading of AI positives (semi-automated);
  - US$66 fully automated;
  - US$77 with human graders only.
  The semi-automated model saved about 20% of screening costs. The fully automated model saved less, reportedly because of more false-positive referrals **(secondary source)**.
- **UK [11].** Automated grading was cost-saving when it replaced the first human grader. The break-even software price was only a few pounds per patient, so the cost being replaced is grader time.

**Illustration: why specificity drives the cost.** This is an illustration, not an economic evaluation.

Assumptions:
- 1,000 people screened.
- 10% have referable DR. This is an assumed community prevalence; APTOS validation is 40%, which is far higher.
- Threshold set at 85% sensitivity.
- Each unnecessary referral costs one optometrist comprehensive diabetic exam, MBS item 10915 at $79.80 [6]. This ignores patient travel, time and any ophthalmology visit, so it underestimates the true cost.

| Model (validation, spec at 85% sens) | Referable found (of 100) | Missed | False positives (of 900) | PPV | FP cost (MBS only) |
|---|---|---|---|---|---|
| Simple colour baseline (spec 0.735) | 85 | 15 | ≈ 238 | 26% | ≈ $19,000 |
| Part A random forest (spec 0.939) | 85 | 15 | ≈ 55 | 61% | ≈ $4,400 |

At the same sensitivity, the better model cuts unnecessary referrals by about three-quarters. The PPV numbers also show that in a real community population, most positives from a weaker model would be false alarms. Specificities come from Part A's validation predictions (`python -m src.metrics`). Replace them with the final models' validation results, never test-set tuning.

**Funding:** the photography service is already funded through MBS items 12325/12326: $59.75 per service, once every 12 months (Aboriginal and Torres Strait Islander patients) or 24 months (others) [5]. There is no separate MBS item for the AI software itself. How the software would be paid for (bundled into the service, a licence paid by the practice, or a new item) is an open question for D's reimbursement section.

## 6. Readiness and what would be needed before clinical use

Our software is a **student prototype**. It is not clinically validated and must not be used for patient diagnosis. The gaps, ordered roughly by importance:

1. **Shortcut learning.** Part A showed that grade 0 photos in APTOS come mainly from different cameras. A model using only image statistics reaches AUROC 0.89 without seeing lesions. We therefore also report grade 1 vs 2–4 AUROC, which is much lower (e.g. 0.86 for A's best model). A deployed system must work for the reason we think it works.
2. **External validation.** All our results come from one dataset (APTOS 2019, India) with no patient IDs. Commercial systems lose performance on new data and cameras:
   - in a US head-to-head test of 7 systems, sensitivity ranged from 51% to 86% [14];
   - in the Australian study, specificity fell to 53% on one camera model [7].
   We would need a prospective study in Australian primary care, including Indigenous patients, on the camera actually deployed.
3. **Reference standard.** The APTOS labels are a single grade per image. We found 30 duplicate groups with conflicting grades. A clinical study needs grading by an agreed protocol (e.g. adjudicated expert grading).
4. **Ungradable images and macular oedema.** Our model has no image-quality output and does not assess diabetic macular oedema, which also needs referral. Both are required for a real screening service.
5. **Benchmark.** IDx-DR, the first FDA-authorised autonomous AI diagnostic system, met pre-specified endpoints of at least 85% sensitivity and 82.5% specificity in a 10-site primary-care trial (87.2% / 90.7%) [15, 16]. That is the level of evidence a clinical claim needs.
6. **Expertise needed:**
   - ophthalmologist or optometrist (reference grading, referral pathway);
   - primary-care and Aboriginal health partners (workflow, cultural safety, community acceptance);
   - health economist (local cost-effectiveness);
   - regulatory affairs (TGA inclusion in the ARTG);
   - clinical IT (integration with GP software and the camera).

---

## References

1. Australian Institute of Health and Welfare. Diabetes: Australian facts — How common is diabetes? (updated 27 May 2026). https://www.aihw.gov.au/reports/diabetes/diabetes-australian-facts/contents/how-common-is-diabetes/all-diabetes
2. Keel S, Xie J, Foreman J, van Wijngaarden P, Taylor HR, Dirani M. The prevalence of diabetic retinopathy in Australian adults with self-reported diabetes: the National Eye Health Survey. *Ophthalmology* 2017;124(7):977–984. doi:10.1016/j.ophtha.2017.02.004
3. Foreman J, Keel S, Xie J, van Wijngaarden P, Taylor HR, Dirani M. Adherence to diabetic eye examination guidelines in Australia: the National Eye Health Survey. *Med J Aust* 2017;206(9):402–406. doi:10.5694/mja16.00989
4. Australian Institute of Health and Welfare. Eye health measures 2025 — Screening for retinopathy among people with self-reported diabetes. https://www.aihw.gov.au/reports/indigenous-australians/eye-health-measures-data-2025/contents/diagnosis-and-screening/screening-retinopathy-with-self-reported-diabetes
5. Australian Government Department of Health. MBS Online, items 12325 and 12326 and explanatory note DN.1.19. https://www9.health.gov.au/mbs/fullDisplay.cfm?type=item&q=12325&qt=item
6. Australian Government Department of Health. MBS Online, item 10915. https://www9.health.gov.au/mbs/fullDisplay.cfm?type=item&q=10915&qt=item
7. Scheetz J, Koca D, McGuinness M, et al. Real-world artificial intelligence-based opportunistic screening for diabetic retinopathy in endocrinology and indigenous healthcare settings in Australia. *Sci Rep* 2021;11:15808. doi:10.1038/s41598-021-94178-5
8. Xie Y, Nguyen QD, Hamzah H, et al. Artificial intelligence for teleophthalmology-based diabetic retinopathy screening in a national programme: an economic analysis modelling study. *Lancet Digit Health* 2020;2(5):e240–e249. doi:10.1016/S2589-7500(20)30060-1
9. Hu W, Joseph S, Li R, et al. Population impact and cost-effectiveness of artificial intelligence-based diabetic retinopathy screening in people living with diabetes in Australia: a cost effectiveness analysis. *eClinicalMedicine* 2024;67:102387. doi:10.1016/j.eclinm.2023.102387
10. Therapeutic Goods Administration. Classifying active medical devices in Australia, including software-based medical devices. https://www.tga.gov.au/resources/guidance/classifying-active-medical-devices-australia-including-software-based-medical-devices *(page could not be opened directly; check the rule wording)*
11. Tufail A, Rudisill C, Egan C, et al. Automated diabetic retinopathy image assessment software: diagnostic accuracy and cost-effectiveness compared with human graders. *Ophthalmology* 2017;124(3):343–351.
12. mivision. Optometrist Warehouse diabetic retinopathy AI study (June 2023). https://mivision.com.au/2023/06/optometrist-warehouse-diabetic-retinopathy-ai-study/ *(news report)*
13. Beede E, Baylor E, Hersch F, et al. A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy. *Proc CHI* 2020. doi:10.1145/3313831.3376718
14. Lee AY, Yanagihara RT, Lee CS, et al. Multicenter, head-to-head, real-world validation study of seven automated artificial intelligence diabetic retinopathy screening systems. *Diabetes Care* 2021;44(5):1168–1175. doi:10.2337/dc20-1877
15. Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digit Med* 2018;1:39. doi:10.1038/s41746-018-0040-6
16. U.S. Food and Drug Administration. De Novo classification request for IDx-DR (DEN180001). https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
