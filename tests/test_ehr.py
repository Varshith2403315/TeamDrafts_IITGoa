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
