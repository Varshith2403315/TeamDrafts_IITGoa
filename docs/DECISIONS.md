# Decision log

| # | Decision | Alternatives | Reason | Trade-off |
|---|---|---|---|---|
| 1 | Build an insulin-dosing twin for T1D on pens | Perioperative hypotension twin on VitalDB; heart-failure twin; dengue twin | VitalDB/PhysioNet unreachable from the build environment; T1D has an open, published simulator; dosing = "personalise treatment", hypo = "adverse event" | Crowded glucose lane; virtual patients only |
| 2 | Hidden-truth design: UVA/Padova truth, Bergman-type twin | Validate the twin on data it generated itself | Different model families avoid a simulator validating itself | Twin is structurally mis-specified on purpose (realistic) |
| 3 | Re-implement simglucose vectorised (RK4) | Use simglucose directly | ~30x faster; enables paired multi-arm trials | Must prove parity -> test within 0.5 mg/dL |
| 4 | Harsher behaviour (carb error 30%, mis-titrated clinic settings, night rescue only on alarm) | Default simglucose scenarios | Default cohort had 83% TIR / 1.4% TBR, implausibly good | Still easier than real life (~76% TIR); stated as a limitation |
| 5 | Fit on 5-h horizons, prior strength 150 | 3-h horizon, weaker prior | 3-h fits recovered CR/CF worse than the clinic record (dev cohort) | Chosen on dev cohort; final numbers from held-out cohorts |
| 6 | Safety margin 12 mg/dL/h capped at 25 | Uncapped margin | Uncapped margin under-dosed (TIR -9 pts on dev) | Small residual hypo risk, covered by alerts |
| 7 | Hybrid alert (twin + CGM trend) | Twin-only alert | Twin-only alert lost to trend extrapolation at matched false-alarm rates despite lower RMSE | Hybrid is less "pure" but measurably better |
| 8 | Separate dev (seed 7) and held-out (2026-2028) cohorts | Report dev results | Dev gain (+5.5 TIR vs premium CGM) shrank to +2.3 on held-out; reporting dev would overclaim | Less impressive headline, more credible |
| 9 | Headline vs threshold-alarm CGM, report premium comparison alongside | Headline vs premium only | Affordable sensors in India lack predictive alarms (Abbott); that is the realistic standard of care | Must show both comparators to avoid cherry-picking |
| 10 | Stop optimising meal dosing | Continue tuning advisor | Oracle analyses: true CR/CF gave lower TIR than clinic settings; perfect bolus adherence adds only ~2.2 TIR -> little headroom | Dosing claim stays modest |
| 11 | Static hosted demo with JS port of the twin | Streamlit / server app | Judges can click and use it with nothing installed; parity-tested | Demo uses pre-computed states for featured meals |
| 12 | Featured demo meals chosen by a fixed rule and disclosed | Hand-picked examples | Avoids silent cherry-picking | Examples are still illustrative |
| 13 | Synthetic EHR with diagnoses, labs, genetics; HbA1c feeds the twin prior (ADAG) | Keep a minimal record | Official brief asks for diagnoses, labs and genetic markers; HbA1c is the lab with a known glucose relation | Other fields are clinical context only (the simulator has no renal/thyroid physiology); stated in the README |
| 14 | Clinician dashboard with settings replay on the virtual patient | Patient view only | Official brief requires a conceptual doctor-facing dashboard | Replay is a twin prediction, labelled as such |
| 15 | Twin settings suggestions capped at +/-20% per review, low confidence if >2x disagreement | Show raw twin estimates | Raw estimates were sometimes absurd (basal 0 U); stepwise titration is standard practice | Smaller suggested changes |
| 16 | After adding HbA1c, re-ran the full study and replaced every reported number | Keep earlier numbers | Results must match the shipped code | The personalisation claim weakened (lows now equal); reported as such |
