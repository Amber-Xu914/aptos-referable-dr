# Role D: clinical and translation research notes

BIOM9027 group project, referable diabetic retinopathy (DR) from APTOS 2019 fundus photos.

These are **research notes for Mooni (role D)**, gathered with AI help. Check each source before using it, and write the slides and the individual report in your own words. Any line marked *(interpretation)* is our reasoning, not a quote from a source.

---

## 1. Unmet need (why this matters in Australia)

- There are more than **1.36 million** Australians with diabetes (Diabetes Australia, Oct 2020).
- Diabetes is the **leading cause of blindness in working-age Australians**. Up to **98%** of serious diabetes-related vision loss can be prevented with regular eye checks and early treatment (Diabetes Australia, 2020).
- Many people miss their eye checks:
  - **National Eye Health Survey** (MJA 2017): **77.5%** of non-Indigenous and **52.7%** of Indigenous Australians with diabetes followed the NHMRC eye-exam guidelines.
  - **Large NSW study** (about 25,000 people, University of Sydney, reported 2020): only about **half** had the recommended eye checks.
- **Recommended interval:** every **2 years** for non-Indigenous Australians and **every year** for Aboriginal and Torres Strait Islander people (NHMRC, as used in the MJA 2017 survey).
- *(interpretation)* **The gap is access and follow-through, not treatment.** A fast, automated first read in the places people already visit (GP clinics, Aboriginal Medical Services, endocrinology clinics) could raise screening coverage.

## 2. Intended use and draft PICO (for the group to agree)

**Intended use (draft):** software that reads one colour fundus photo from an adult with diabetes and tells the clinician whether the photo shows **referable DR (APTOS grade 2 or higher)**. The clinician makes the final decision. It is a triage and decision-support tool, not a diagnosis.

**"If the model is positive, the clinician will…"** refer the patient to an optometrist or ophthalmologist for a full eye examination.

| | |
|---|---|
| **P** | Adults with diabetes and no known DR, seen in primary care, Aboriginal Medical Services or endocrinology clinics, with one non-mydriatic colour fundus photo |
| **I** | Our model's referable / non-referable output (threshold chosen on validation for at least 90% sensitivity) |
| **C** | **Comparator:** current pathway (GP/optometrist grading of the photo, or referral of everyone to an eye-care professional). **Reference standard:** APTOS clinician grade on the 0–4 scale (single image, single grade). Keep these two separate. |
| **O** | Primary: sensitivity for referable DR. Secondary: specificity, PPV/NPV, referral rate, proportion of ungradable images |

**Why the threshold favours sensitivity:**
- A **false negative** means referable DR is missed. The patient may not be checked again for 1–2 years, and the disease can progress to vision loss.
- A **false positive** means an unnecessary referral. It costs a specialist visit, money and some anxiety, but it is recoverable.

## 3. Workflow (where the software fits)

1. A nurse or practice staff member takes the fundus photo with a non-mydriatic camera, ideally during the routine diabetes visit.
2. **Image-quality check:** ungradable photos are referred, not graded. The optometry guidance on MBS retinal photography also says inadequate images should lead to referral.
3. The software gives its output to the GP. With the two-threshold triage design, the uncertain band goes to a human grader.
4. If the result is referable, the patient is referred to an optometrist or ophthalmologist. If not, the patient is recalled at the normal interval (1 or 2 years).
5. *(interpretation)* **Referral follow-through is the weak link.** In the Australian real-world AI study (section 6), only 60% of the contacted patients who were referred attended. A positive result only helps if the referral actually happens.

## 4. Regulation (TGA)

- Software that diagnoses or screens for a disease is a medical device in Australia and needs **ARTG inclusion** before supply.
- **Classification rule 4.5** (TGA guidance on software-based devices) depends on two things: who makes the decision, and how serious the condition is.
  - **Device makes the decision itself, serious disease:** Class **IIb** (4.5(1)(d))
  - **Device informs a health professional who decides, serious disease:** Class **IIa** (4.5(2)(b))
  - The TGA defines "serious disease" as one that may cause long-term disability, among other criteria.
- *(interpretation)* **DR can cause blindness, so it fits "serious disease".**
  - Our intended use (the clinician decides) would most likely be **Class IIa**.
  - A fully autonomous version, like IDx-DR in the US, would be **Class IIb**.
  - This needs confirming. It is our reading of the guidance, not a TGA decision.
- **Other requirements to mention:**
  - quality management system (ISO 13485)
  - clinical evidence
  - cybersecurity
  - a plan for **changes after approval** (retraining the model counts as a change to the device)
  - post-market monitoring

## 5. Reimbursement (Medicare)

- **MBS items** for bilateral retinal photography with a non-mydriatic camera in people with diabetes, introduced in **November 2016**:
  - **12325:** Aboriginal and Torres Strait Islander people, claimable every **12 months**
  - **12326:** other Australians, claimable every **2 years**
- These items are claimed by the medical practitioner coordinating the patient's care, who keeps overall responsibility.
- *(interpretation)*
  - We found no separate MBS item for AI grading.
  - An AI tool would most likely be funded **inside** the existing retinal-photography visit, as a cost to the practice.
  - A new item would need an **MSAC** application. The original retinal-photography item came through MSAC application 1181.
- **Cost-benefit (qualitative):**
  - **Benefits:** fewer unnecessary specialist visits, more people screened where they already are, earlier treatment, and avoided vision loss (a large long-term cost).
  - **Costs:** camera, software licence, staff time, and the specialist visits created by false positives.
  - We have no cost data of our own. The Australian AI study also did **no cost-effectiveness analysis**, which is a gap worth mentioning.

## 6. Evidence benchmarks (what "good enough" looked like for real products)

**IDx-DR pivotal trial** (Abràmoff et al., *npj Digital Medicine* 2018), the first autonomous AI diagnostic authorised by the US FDA (2018):
- **900** people with diabetes, recruited in primary care
- **Endpoint:** more-than-mild DR or macular oedema
- **Results:** sensitivity **87.2%**, specificity **90.7%**, imageability **96.1%**
- **Reference standard:** Wisconsin Fundus Photograph Reading Center grading of widefield stereo photos plus macular OCT, which is **much stronger than our single-image grade**

**Australian real-world study** (Scheetz et al., *Scientific Reports* 2021):
- **Setting:** 2 endocrinology clinics in Melbourne and 3 Aboriginal Medical Service clinics in Western Australia
- **Participants:** 236 people screened, **203** analysed
- **Results for referable DR:** sensitivity **96.9%** (95% CI 83.8–99.9), specificity **87.7%** (95% CI 81.8–92.2), AUC 0.92
- **14% of participants had ungradable images** (small pupils, cataract and other reasons)
- **Specificity dropped:**
  - to **70%** in the Aboriginal Medical Service clinics
  - to **53%** with one camera model
- The algorithm had been trained only on images from **Chinese adults**
- **93.7%** of patients were satisfied
- Only 60% of the contacted patients who were referred attended

**Lessons for our project** *(interpretation)*:
- Image quality matters.
- The camera and the population matter (generalisability).
- Strong offline numbers do not guarantee real-world performance.

## 7. Limitations specific to our data (from the group repo)

- **Single source:** APTOS images come from Aravind Eye Hospital in India. They include **17 different image sizes** from different cameras, and many photos have the top and bottom of the field of view cut off. The data are not Australian, and Aboriginal and Torres Strait Islander people are not represented.
- **No patient IDs:** we can't rule out two photos of the same patient sitting in different splits. Byte-identical duplicates *are* kept in the same split.
- **Noisy reference standard:**
  - 123 groups of identical images (251 images) are in the dataset.
  - In **30 groups**, copies of the same image have different grades.
  - In **6 groups**, the same image is labelled both referable and non-referable.
  - **18 test images** belong to a conflicting group.
  - So even a perfect model cannot reach 100% against these labels.
- **Single image, single grade:** there is no multi-grader adjudication and no OCT. The data also contain no separate macular oedema label.
- **Disease-enriched data:** about **40%** of APTOS images are referable, which is much higher than in real screening.
  - PPV depends on prevalence. With 90% sensitivity and 90% specificity, PPV is about **86% at 40% prevalence** but only about **32% at 5% prevalence**. The 5% figure is illustrative, not a sourced prevalence.
  - So in practice, most positives would be false referrals, and the real-world PPV would be much lower than our test-set PPV.

## 8. What is still needed before clinical use

1. **External validation** on Australian images: different cameras, GP and Aboriginal Medical Service settings, and Indigenous patients.
2. A **prospective study** in the intended setting, against a stronger reference standard (multiple graders or OCT).
3. An **image-quality gate** and a clear pathway for ungradable images.
4. **Calibration** and threshold checks on the new population.
5. **Expertise needed:**
   - ophthalmologists and optometrists (reference standard, safety)
   - GPs and practice nurses (workflow, usability)
   - Aboriginal community-controlled health services (cultural safety, acceptability)
   - regulatory and quality specialists (TGA)
   - health economists (cost-effectiveness for MSAC)
6. A **TGA submission** (likely Class IIa) and a plan for monitoring and updating the model.

## 9. Concluding question (rubric): template for the final slide

> **Technically reliable?** On held-out APTOS images, [best model] reached AUROC [x] (95% CI …), with sensitivity [x] and specificity [x] at a threshold chosen on validation. Errors concentrate in [grade 1/2 boundary, poor-quality photos, label-conflict images…].
>
> **Medically valuable?** Potentially, as a triage tool that raises screening coverage in primary care.
>
> **What evidence is still required?** External, prospective validation in Australian settings with a stronger reference standard, plus TGA approval and a funding pathway.
>
> **This software is not clinically validated.**

---

## Sources

- Diabetes Australia (2020). *50 percent of people living with diabetes are missing out on diabetes eye checks.* https://www.diabetesaustralia.com.au/mediarelease/50-percent-of-people-living-with-diabetes-are-missing-out-on-diabetes-eye-checks-to-prevent-blindness/
- Foreman J, et al. (2017). Adherence to diabetic eye examination guidelines in Australia: the National Eye Health Survey. *Med J Aust* 206(9). https://www.mja.com.au/journal/2017/206/9/adherence-diabetic-eye-examination-guidelines-australia-national-eye-health
- Optometry Australia. MBS items 12325/12326 retinal photography. https://www.optometry.org.au/?p=8666
- MSAC Application 1181, non-mydriatic retinal photography. https://msac.gov.au/sites/default/files/documents/1181-FinalPSD-Non-mydriaticretinalphotography-accessible.pdf
- TGA. Classification of active medical devices (including software-based medical devices). https://www.tga.gov.au/sites/default/files/classification-of-active-medical-devices-including-software-based-medical-devices.docx
- Abràmoff MD, et al. (2018). Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digit Med* 1:39. https://pmc.ncbi.nlm.nih.gov/articles/PMC6550188/ (figures checked via https://www.healio.com/news/optometry/20180910/ai-diagnostic-system-accuracy-holds-up-to-industry-standards)
- Scheetz J, et al. (2021). Real-world artificial intelligence-based opportunistic screening for diabetic retinopathy in endocrinology and indigenous healthcare settings in Australia. *Sci Rep* 11. https://pmc.ncbi.nlm.nih.gov/articles/PMC8339059/
- Dataset: acknowledge the Asia Pacific Tele-Ophthalmology Society (APTOS), Aravind Eye Hospital and Kaggle for the APTOS 2019 Blindness Detection dataset.
