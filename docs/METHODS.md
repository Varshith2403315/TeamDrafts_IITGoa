# Methods

## 1. Ground truth (hidden from the twin)
UVA/Padova T1DM model, 13 states, parameters for 30 virtual patients (10 children, 10 adolescents, 10 adults) from `simglucose` 0.2.11 (MIT). Integrated with RK4 (two 30-s sub-steps per minute), vectorised across patients. Parity test vs `simglucose.T1DPatient`: max deviation < 0.5 mg/dL over 10 h with a meal and bolus (`tests/test_truth_parity.py`).

Extensions (truth side only): insulin-dependent utilisation multiplied by a daily sensitivity factor (log-normal, SD 0.18) and by (1 + gain x activity state) during and after walks (gain 0.6-1.6, onset tau 20 min, offset tau 90 min); endogenous production raised up to 35% around 06:00 (dawn).

## 2. Life simulation (`dosetwin/truth/cohort.py`)
14 days per patient, 1-min resolution. Meals: breakfast ~08:00 (40-80 g), lunch ~13:30 (70-120 g), snack ~17:30 (20-40 g, 70% of days, 40% bolused), dinner ~21:00 (70-130 g). Portions scaled 0.55 (children), 0.85 (adolescents). Carb estimate = true x personal bias U(0.65, 1.10) x lognormal(0, 0.3), rounded to 5 g. Boluses: 10% missed, 12% given 30 min late. Clinic settings: CR and CF = ideal x U(0.75, 1.30); long-acting basal = physiological x U(0.8, 1.2), delivered as a flat rate. Standard calculator: carbs/CR + max(0, (CGM-120)/CF) - IOB (linear 4 h), rounded to 0.5 U. Rescue: 15 g if CGM < 70 awake (< 55 asleep, alarm), at most every 20 min. Corrections: awake, CGM > 250, IOB < 0.5 U, at most every 3 h. CGM: Dexcom noise model from simglucose, 5-min samples. Steps: ~110/min while walking + daytime background.

## 3. Twin
State [S1, S2, I, X, D1, D2, E, G, Gs, d]:
- S1' = u + b - S1/ti; S2' = (S1-S2)/ti; I' = S2/ti - ke I (ke = 0.138/min)
- X' = -p2 X + p2 SI (I - b/ke)/W (p2 = 0.02/min)
- D1' = c - D1/tm; D2' = (D1-D2)/tm; Ra = fc D2/tm x 1000/(1.6 W)
- E' = (a(steps) - E)/60
- G' = -SG (G-Gb) - X G - alpha E G/1000 + Ra + dawn x exp(-((t mod 1440 - 360)/60)^2/2) + d
- Gs' = (G - Gs)/8; in forecasts d decays with tau 30 min.

Parameters theta = (log SI, Gb, log fc, log tm, log ti, alpha, dawn, log SG). Explicit Euler at 1 min.

**Prior from the EHR:** fasting-glucose prior Gb = ADAG mean glucose from HbA1c (28.7 x A1c - 46.7) minus 10 mg/dL, clipped to 80-220. SI is chosen so that the twin's 5-h drop after 1 U equals the prescribed CF. fc is chosen so that a 50 g meal covered by 50/CR units returns to baseline at 5 h. The other parameters take population defaults.

**MAP fit:** windows every 30 min over days 1-7; open-loop 5-300 min forecasts with known inputs (identification only); soft-L1 loss; Gaussian prior penalty with SDs (0.5, 25, 0.4, 0.35, 0.3, 2.5, 0.15, 0.6); bounded trust-region least squares (scipy). Median 1.3 s per patient.

**Observer:** at each CGM sample, Gs += 0.6 e, G += 0.5 e, d += 0.004 e (d clipped to +/-3 mg/dL/min), where e = CGM - Gs.

## 4. Dose advisor
Candidate doses 0 .. max(2 x calculator dose, calculator + 3 U) in 0.5 U steps. Each is simulated 300 min from the synchronised state with the logged carbs. Feasible if min(forecast - min(12 t/60, 25)) >= 70. Choose the minimum mean Kovatchev risk among feasible doses; if none is feasible, choose 0 U.

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
- D: DoseTwin (twin dosing + hybrid alert)
- E: D with the EHR-prior twin (no personalisation)

Metrics per consensus (Battelino 2019). Paired per-patient differences averaged over seeds (n = 30); 5,000-sample bootstrap CI; Wilcoxon signed-rank.

## 7. Baseline
LightGBM (400 trees) on 13 CGM lags, 3 deltas, IOB, COB, steps over 30/60 min, time of day and EHR settings. It predicts glucose change at 30/60/120 min, plus a classifier for hypo onset within 60 min. Trained on dev-cohort true glucose labels, which is a privileged advantage over the twin.

## 8. Synthetic EHR
`dosetwin/ehr.py`: demographics (age, sex, height, weight, BMI, state), ICD-10 diagnoses (T1D with onset age; autoimmune hypothyroidism 15%; coeliac 6%; severe hypoglycaemia in last 12 months 20%; retinopathy in some adults), labs (HbA1c = ADAG inverse of mean glucose over the 7-day run-in + N(0, 0.25); C-peptide, eGFR, TSH, LDL, urine ACR), genetics (HLA-DR genotype, GAD65 status), prescriptions. Seeded per patient.

## 9. Clinician dashboard
Patients ranked by 2 x time-below-70 + night time-below-70 + (100 - TIR)/10 over the run-in week (CGM). Twin settings suggestion = current x clip(twin/current, 0.8, 1.2); flagged low-confidence (no change) if twin/current is outside (0.5, 2). Settings replay: day 7 of the run-in simulated on the twin from its synchronised midnight state, with logged meals and steps, calculator boluses recomputed with the trial CR/CF, basal change applied as an extra infusion, rule-of-15 rescue below 70.
