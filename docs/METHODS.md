# Methods

## 1. Ground truth (hidden from the twin)
UVA/Padova T1DM model, 13 states, parameters for 30 virtual patients (10 children, 10 adolescents, 10 adults) from `simglucose` 0.2.11 (MIT). Integrated with RK4 (two 30-s sub-steps per minute), vectorised across patients. Parity test vs `simglucose.T1DPatient`: max deviation < 0.5 mg/dL over 10 h with a meal and bolus (`tests/test_truth_parity.py`).

Scope: the model is built for insulin-treated diabetes and validated here on type 1 (the only open, published simulator with virtual patients). Type 2 diabetes on basal-bolus insulin is the next validation step.

Extensions (truth side only; with all of them switched off the simulator reproduces simglucose):
- insulin-dependent utilisation multiplied by a residual daily sensitivity factor (log-normal, SD 0.15), by (1 + gain x activity state) during and after exercise (gain 0.6-1.6, onset tau 20 min, offset tau 90 min), and by exp(-k x sleep debt) for the day after each night (k = 0.06 x U(0.5, 1.5) per hour; about 20% lower sensitivity after a 4-h night, in line with Donga et al., Diabetes Care 2010);
- endogenous production raised up to 35% around 06:00 (dawn) and by counter-regulation during hypoglycaemia: x (1 + c x clip((75 - G)/35, 0, 1)), c = 0.5, or 0.1 with impaired hypoglycaemia awareness;
- insulin degradation (m4 and m30) multiplied by 1 - 0.2 x clip((90 - eGFR)/60, 0, 1): 1 at eGFR >= 90, 0.8 at eGFR 30.

## 2. Life simulation (`dosetwin/truth/cohort.py`)
14 days per patient, 1-min resolution. Meals: breakfast ~08:00 (40-80 g), lunch ~13:30 (70-120 g), snack ~17:30 (20-40 g, 70% of days, 40% bolused), dinner ~21:00 (70-130 g). Portions scaled 0.55 (children), 0.85 (adolescents). Carb estimate = true x personal bias U(0.65, 1.10) x lognormal(0, 0.3), rounded to 5 g. Boluses: 10% missed, 12% given 30 min late. Clinic settings: CR and CF = ideal x U(0.75, 1.30); long-acting basal = physiological x U(0.8, 1.2), delivered as a flat rate. Standard calculator: carbs/CR + max(0, (CGM-120)/CF) - IOB (linear 4 h), rounded to 0.5 U. Rescue: 15 g if CGM < 70 awake (< 55 asleep, alarm), at most every 20 min. Corrections: awake, CGM > 250, IOB < 0.5 U, at most every 3 h. CGM: Dexcom noise model from simglucose, 5-min samples. Steps: ~110/min while walking + daytime background.

Exercise: evening walks on 50% of days (seen by steps) and other exercise with few steps (cycling, gym, yoga) on 30% of days, intensity 0.5-0.9 (seen only by heart rate). Heart rate per minute = resting HR (58-90 bpm by age) + 10 awake + 70 x exercise intensity + N(0, 3).

Sleep: onset ~23:30 (SD 30 min, 23:00-01:30), duration N(personal mean U(6.0, 7.8) h, 0.7 h), 15% of nights short (3.5-5.5 h), awakenings Poisson(1.5), wake-up no later than 07:45. Sleep debt = max(0, 7 - hours) + 0.5 x max(0, awakenings - 2), in force from wake-up to the next wake-up. The wearable reports hours with error N(0, 0.3 h) and awakenings within +/-1. People are "asleep" (rescue only on the < 55 alarm) between their actual onset and wake-up.

Kidney function and awareness (enriched so the effects are measurable, not representative): adults eGFR U(30, 60) with probability 0.35, otherwise N(95, 14); children and adolescents U(45, 75) with probability 0.08, otherwise N(110, 12). Impaired hypoglycaemia awareness in 25% of patients.

## 3. Twin
State [S1, S2, I, X, D1, D2, E, G, Gs, d]:
- S1' = u + b - S1/ti; S2' = (S1-S2)/ti; I' = S2/ti - ke I (ke = 0.138/min)
- X' = -p2 X + p2 SI (I - b/ke)/W (p2 = 0.02/min)
- D1' = c - D1/tm; D2' = (D1-D2)/tm; Ra = fc D2/tm x 1000/(1.6 W)
- X' = -p2 X + p2 SI exp(-bs sd) (I - b/ke)/W, where sd is the wearable's sleep debt (h)
- E' = (a(act) - E)/60, a = clip((act - 40)/80, 0, 1)
- G' = -SG (G-Gb) - X G - alpha E G/1000 + Ra + dawn x exp(-((t mod 1440 - 360)/60)^2/2) + d
- Gs' = (G - Gs)/8; in forecasts d decays with tau 30 min.

Parameters theta = (log SI, Gb, log fc, log tm, log ti, alpha, dawn, log SG, bs). Explicit Euler at 1 min.

**Wearable fusion** (`dosetwin/fusion.py`): activity in step-equivalents act = max(steps, 40 + 80 a_hr) when a_hr = clip((HR - HR_rest - 25)/50, 0, 1) > 0, else steps; HR_rest = 10th percentile of the previous day's heart rate.

**Prior from the EHR:** fasting-glucose prior Gb = ADAG mean glucose from HbA1c (28.7 x A1c - 46.7) minus 10 mg/dL, clipped to 80-220. SI is chosen so that the twin's 5-h drop after 1 U equals the prescribed CF, then raised for reduced kidney function: log SI += c_egfr x clip((90 - eGFR)/60, 0, 1). c_egfr is the slope of (fitted - prior) log SI on that term over the 30 development-cohort twins, frozen before the held-out cohorts are touched. fc is chosen so that a 50 g meal covered by 50/CR units returns to baseline at 5 h. bs starts at 0.05 per hour (prior SD 0.04, bounds 0-0.25). The other parameters take population defaults.

**History of severe hypoglycaemia** (ICD-10 E16.0 in the record; present for ~90% of patients with impaired awareness and 4% of others): dose-advisor safety floor 80 instead of 70 mg/dL, low alert and bedtime alert 7 mg/dL earlier. Both values were fixed a priori (less stringent targets for impaired awareness, ADA Standards of Care section 6), not tuned.

**MAP fit:** windows every 30 min over days 1-7; open-loop 5-300 min forecasts with known inputs (identification only); soft-L1 loss; Gaussian prior penalty with SDs (0.5, 25, 0.4, 0.35, 0.3, 2.5, 0.15, 0.6, 0.04); bounded trust-region least squares (scipy). Median about 1.6 s per patient.

**Observer:** at each CGM sample, Gs += 0.6 e, G += 0.5 e, d += 0.004 e (d clipped to +/-3 mg/dL/min), where e = CGM - Gs.

## 4. Dose advisor
Candidate doses 0 .. max(2 x calculator dose, calculator + 3 U) in 0.5 U steps. Each is simulated 300 min from the synchronised state with the logged carbs. Feasible if min(forecast - min(12 t/60, 25)) >= floor (70 mg/dL, or 80 with a history of severe hypoglycaemia). Choose the minimum mean Kovatchev risk among feasible doses. If none is feasible (glucose already low or falling at the meal), choose the lowest-risk dose whose pessimistic minimum is within 5 mg/dL of the no-insulin path, so a large meal is still covered.

## 4b. Bedtime check (`dosetwin/bedtime.py`)
At 23:00 the synchronised twin simulates 8 h with no further food or bolus, twice: nominal, and pessimistic (SI x 1.25). Risk score = minimum of the pessimistic path. Flag if it is below the threshold (frozen on the dev cohort at 80% specificity; +7 mg/dL with a history of severe hypoglycaemia). When flagged, the twin suggests the smallest snack in {10, 15, 20, 25, 30} g whose pessimistic night stays above 80 mg/dL; in the study the patient eats it with probability 0.8.

Evaluation: one row per patient-night for nights starting on days 8-13 (all inside the evaluation week), excluding nights where CGM at 23:00 is already below 70. Event = true glucose < 70 for >= 15 min between 23:00 and 07:00, on standard care (no intervention). Comparators get the same information: the bedtime-glucose rule (CGM at 23:00), and a LightGBM classifier (300 trees) on bedtime CGM, 15/30-min trends, IOB, COB, activity today, sleep debt, the patient's own CGM-judged history of night lows and night glucose, and the EHR fields, trained on dev nights 2-13. A LightGBM + twin-feature model is also reported. AUROC with patient-level bootstrap CIs; thresholds for "caught" frozen on dev at 80% specificity (LightGBM via patient-grouped out-of-fold predictions).

## 5. Alerts
- Trend: least-squares slope over the last 15 min of CGM, projected 30 min ahead.
- Twin: minimum of the 40-min open-loop forecast with no future inputs.
- Hybrid: mean of the two. Alert below 65 mg/dL (trend arm: below 50 mg/dL); both thresholds set at <= 2 false alerts/day on the dev cohort. Response: 80% of alerts lead to 15 g carbs; 30-min refractory period.
- Event: true glucose < 70 for >= 15 min. Hit: an alert within 60 min before onset while glucose >= 70. False alert: no onset within 60 min.

## 6. Study
Dev cohort seed 7: frozen settings and LightGBM training. Held-out seeds 2026, 2027, 2028. Arms (days 8-14, identical exogenous inputs):
- A: calculator + threshold alarms
- B: A + trend alert
- C: twin dosing + trend alert
- D: DoseTwin without the bedtime check (twin dosing + hybrid alert)
- E: D with the EHR-prior twin (no personalisation)
- F: D with the twin refitted without HbA1c, eGFR and the severe-hypo diagnosis (ablation)
- G: DoseTwin (D + bedtime check)

Ablations are refitted, not switched off after fitting: F refits every twin with the labs and diagnosis removed from the prior and from the advisor/alert adjustments; the "without heart rate and sleep" twin is refitted on steps only with no sleep input.

Metrics per consensus (Battelino 2019). Paired per-patient differences averaged over seeds (n = 30); 5,000-sample bootstrap CI; Wilcoxon signed-rank.

## 7. Baseline
LightGBM (400 trees) on 13 CGM lags, 3 deltas, IOB, COB, steps over 30/60 min, mean heart rate over 30 min, sleep debt, time of day and EHR fields (settings, weight, HbA1c, eGFR, severe-hypo history). It predicts glucose change at 30/60/120 min, plus a classifier for hypo onset within 60 min. Trained on dev-cohort true glucose labels, which is a privileged advantage over the twin.

## 8. Synthetic EHR
`dosetwin/ehr.py`: demographics (age, sex, height, weight, BMI, state), ICD-10 diagnoses (T1D with onset age; autoimmune hypothyroidism 15%; coeliac 6%; E16.0 severe hypoglycaemia in last 12 months, tied to the hidden impaired-awareness trait; E10.2 diabetic kidney disease when true eGFR < 60; retinopathy in some adults), labs (HbA1c = ADAG inverse of mean glucose over the 7-day run-in + N(0, 0.25); eGFR = true eGFR + N(0, 4); urine ACR higher with low eGFR; C-peptide, TSH, LDL), genetics (HLA-DR genotype, GAD65 status), prescriptions. Seeded per patient. Thyroid, coeliac, genetics and the other labs have no modelled physiological effect and are not used by the twin.

## 9. Clinician dashboard
Patients ranked by 2 x time-below-70 + night time-below-70 + (100 - TIR)/10 over the run-in week (CGM), + 12 if tonight's bedtime check flags a low. Each patient shows tonight's nominal and pessimistic night forecast, the suggested snack, and a wearable summary (sleep, short nights, sleep debt, resting HR, days with exercise seen only by heart rate). Twin settings suggestion = current x clip(twin/current, 0.8, 1.2); flagged low-confidence (no change) if twin/current is outside (0.5, 2). Settings replay: day 7 of the run-in simulated on the twin from its synchronised midnight state, with logged meals and steps, calculator boluses recomputed with the trial CR/CF, basal change applied as an extra infusion, rule-of-15 rescue below 70.
