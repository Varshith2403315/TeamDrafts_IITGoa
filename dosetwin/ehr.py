"""Synthetic electronic health records (Synthea-style, FHIR-flavoured).

One record per virtual patient: demographics, diagnoses, lab results, genetic
markers and current prescriptions. Records are synthetic (no real patient data,
DPDP/HIPAA-safe) and seeded per patient.

What the twin uses from the record:
  * weight, prescribed carb ratio (CR), correction factor (CF), basal dose
    -> the EHR-informed prior (identify.ehr_prior);
  * HbA1c -> prior on fasting glucose via the ADAG relation
    (estimated average glucose = 28.7 x HbA1c - 46.7; Nathan et al. 2008);
  * eGFR -> prior on insulin potency (reduced kidney function slows insulin
    clearance; coefficient estimated on the development cohort);
  * "history of severe hypoglycaemia" (ICD-10 E16.0) -> a higher safety floor
    for dose advice and an earlier low alert, and a bedtime-risk feature.
In the hidden simulator eGFR and impaired hypoglycaemia awareness have real
physiological effects, and the record reflects them imperfectly (lab noise;
the diagnosis is recorded for about 90% of patients with impaired awareness
and 4% of others). Diagnoses other than E16.0, the remaining labs and genetics
are shown to the clinician but have no modelled effect, so they are not used
by the algorithm and not claimed as evaluated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["Karnataka", "Goa", "Maharashtra", "Tamil Nadu", "Kerala", "Telangana", "Delhi", "West Bengal",
          "Gujarat", "Uttar Pradesh"]


def hba1c_from_mean_glucose(mean_bg: float) -> float:
    return (mean_bg + 46.7) / 28.7


def adag_mean_glucose(hba1c: float) -> float:
    return 28.7 * hba1c - 46.7


def build_records(world, log, seed: int, history_days: int = 7) -> list[dict]:
    """HbA1c reflects the pre-evaluation period (days 1-7) plus lab noise."""
    ehr = world.ehr()
    out = []
    for i, n in enumerate(world.names):
        rng = np.random.RandomState(seed * 4099 + i)
        e = ehr.iloc[i]
        age = int(e.age)
        sex = "female" if rng.rand() < 0.5 else "male"
        height = float(np.clip(rng.normal(120 + 3.2 * min(age, 18) if age < 18 else (171 if sex == "male" else 157), 7), 110, 195))
        bmi = float(e.weight_kg) / (height / 100) ** 2
        mean_bg = float(np.nanmean(log["bg"][i, :history_days * 1440]))
        a1c = round(hba1c_from_mean_glucose(mean_bg) + rng.normal(0, 0.25), 1)
        onset_age = int(np.clip(rng.uniform(3, min(age, 30)), 1, age))
        conditions = [{"code": "E10", "display": "Type 1 diabetes mellitus", "onset_age": onset_age}]
        if rng.rand() < 0.15:
            conditions.append({"code": "E03.9", "display": "Hypothyroidism (autoimmune)"})
        if rng.rand() < 0.06:
            conditions.append({"code": "K90.0", "display": "Coeliac disease"})
        beh = world.behaviours[i]
        if rng.rand() < (0.9 if beh.impaired_awareness else 0.04):
            conditions.append({"code": "E16.0", "display": "History of severe hypoglycaemia (past 12 months)"})
        if beh.egfr < 60:
            conditions.append({"code": "E10.2", "display": "Type 1 diabetes with kidney complications (diabetic kidney disease)"})
        if age >= 18 and age - onset_age > 10 and rng.rand() < 0.25:
            conditions.append({"code": "E10.3", "display": "Background diabetic retinopathy"})
        labs = {
            "hba1c_pct": a1c,
            "fasting_c_peptide_nmol_l": round(float(rng.uniform(0.01, 0.15)), 2),
            "egfr_ml_min": int(np.clip(beh.egfr + rng.normal(0, 4), 15, 150)),
            "tsh_miu_l": round(float(rng.lognormal(np.log(2.2), 0.35) * (1.8 if any(c["code"] == "E03.9" for c in conditions) else 1)), 1),
            "ldl_mg_dl": int(np.clip(rng.normal(95 if age < 18 else 110, 22), 50, 190)),
            "urine_acr_mg_g": int(np.clip(rng.lognormal(np.log(12 if beh.egfr >= 60 else 60), 0.6), 2, 600)),
        }
        hla = rng.choice(["DR3/DR4", "DR3/DR3", "DR4/DR4", "DR3/X", "DR4/X", "X/X"], p=[.32, .14, .14, .16, .16, .08])
        genetics = {"hla_dr_genotype": str(hla),
                    "gad65_antibody_at_diagnosis": "positive" if rng.rand() < 0.75 else "negative"}
        meds = [
            {"drug": "Insulin glargine", "dose": f"{e.basal_u_per_day:.0f} U once daily"},
            {"drug": "Insulin aspart (pen)", "dose": f"1 U per {e.carb_ratio_g_per_u:.0f} g carbohydrate; "
                                                   f"1 U lowers glucose {e.correction_factor_mgdl_per_u:.0f} mg/dL"},
        ]
        if any(c["code"] == "E03.9" for c in conditions):
            meds.append({"drug": "Levothyroxine", "dose": f"{int(rng.choice([25, 50, 75]))} mcg daily"})
        out.append({
            "patient_id": n,
            "demographics": {"age": age, "sex": sex, "height_cm": round(height), "weight_kg": float(e.weight_kg),
                             "bmi": round(bmi, 1), "state": str(rng.choice(STATES))},
            "conditions": conditions,
            "labs": labs,
            "genetics": genetics,
            "prescriptions": {"carb_ratio_g_per_u": float(e.carb_ratio_g_per_u),
                              "correction_factor_mgdl_per_u": float(e.correction_factor_mgdl_per_u),
                              "basal_u_per_day": float(e.basal_u_per_day),
                              "total_daily_insulin_u": float(e.total_daily_insulin_u)},
            "medications": meds,
        })
    return out


def ehr_table(world, log, seed: int) -> pd.DataFrame:
    """The model-facing columns (ehr DataFrame + HbA1c)."""
    recs = build_records(world, log, seed)
    df = world.ehr()
    df["hba1c_pct"] = [r["labs"]["hba1c_pct"] for r in recs]
    df["egfr_ml_min"] = [r["labs"]["egfr_ml_min"] for r in recs]
    df["severe_hypo_history"] = [int(any(c["code"] == "E16.0" for c in r["conditions"])) for r in recs]
    return df
