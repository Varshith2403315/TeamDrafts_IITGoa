import numpy as np

from dosetwin.ehr import adag_mean_glucose, build_records, hba1c_from_mean_glucose
from dosetwin.truth.cohort import World
from dosetwin.twin import identify as I


def test_adag_roundtrip():
    assert abs(adag_mean_glucose(hba1c_from_mean_glucose(154.0)) - 154.0) < 1e-9
    assert abs(adag_mean_glucose(7.0) - 154.2) < 0.1  # ADAG: HbA1c 7% ~ 154 mg/dL


def test_records_have_required_fields_and_drive_prior():
    w = World(["child#001", "adult#003"], days=2, seed=1)
    log = {"bg": np.full((2, 2 * 1440), 160.0)}
    recs = build_records(w, log, seed=1, history_days=2)
    for r in recs:
        assert {"demographics", "conditions", "labs", "genetics", "prescriptions", "medications"} <= set(r)
        assert r["conditions"][0]["code"] == "E10"
        assert 5.5 < r["labs"]["hba1c_pct"] < 9.0
    ehr = w.ehr(); ehr["hba1c_pct"] = [6.0, 9.0]
    lo = I.ehr_prior(I.from_log({"bolus": np.zeros((2, 10)), "carbs_logged": np.zeros((2, 10)), "steps": np.zeros((2, 10)),
                                 "cgm": np.zeros((2, 10))}, ehr, 0))
    hi = I.ehr_prior(I.from_log({"bolus": np.zeros((2, 10)), "carbs_logged": np.zeros((2, 10)), "steps": np.zeros((2, 10)),
                                 "cgm": np.zeros((2, 10))}, ehr, 1))
    assert hi[1] > lo[1]  # higher HbA1c -> higher fasting-glucose prior


def test_record_reflects_hidden_kidney_function_and_awareness():
    names = [f"adult#{k:03d}" for k in range(1, 11)] + [f"child#{k:03d}" for k in range(1, 11)]
    w = World(names, days=1, seed=3)
    recs = build_records(w, {"bg": np.full((20, 1440), 150.0)}, seed=3, history_days=1)
    for b, r in zip(w.behaviours, recs):
        assert abs(r["labs"]["egfr_ml_min"] - b.egfr) < 20
        if b.egfr < 60:
            assert any(c["code"] == "E10.2" for c in r["conditions"])
    assert any(b.clearance < 1 for b in w.behaviours)


def test_egfr_raises_insulin_potency_prior():
    w = World(["adult#001", "adult#002"], days=1, seed=1)
    ehr = w.ehr(); ehr["egfr_ml_min"] = [100.0, 35.0]
    ehr["carb_ratio_g_per_u"] = 10.0; ehr["correction_factor_mgdl_per_u"] = 40.0; ehr["basal_u_per_day"] = 14.4
    ehr["weight_kg"] = 70.0
    log = {"bolus": np.zeros((2, 10)), "carbs_logged": np.zeros((2, 10)), "steps": np.zeros((2, 10)),
           "cgm": np.zeros((2, 10))}
    a = I.ehr_prior(I.from_log(log, ehr, 0), egfr_coef=0.4)
    b = I.ehr_prior(I.from_log(log, ehr, 1), egfr_coef=0.4)
    assert b[0] > a[0]
    c = I.ehr_prior(I.from_log(log, ehr, 1, use_ehr_labs=False), egfr_coef=0.4)
    assert abs(c[0] - a[0]) < 1e-6  # ablation ignores the lab
