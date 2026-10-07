"""Our vectorised UVA/Padova must reproduce the original simglucose patient."""
import importlib.util
import sys
import types

import numpy as np
import pytest

from dosetwin.truth.uva_padova import PatientBatch, load_params


_SG = None


def _load_simglucose_patient():
    global _SG
    if _SG is not None:
        return _SG
    # simglucose's package __init__ imports gym; load the patient module directly
    spec = importlib.util.find_spec("simglucose")
    if spec is None:
        pytest.skip("simglucose not installed")
    pkgdir = spec.submodule_search_locations[0]
    for name in ["simglucose", "simglucose.patient"]:
        m = types.ModuleType(name)
        m.__path__ = [pkgdir if name == "simglucose" else pkgdir + "/patient"]
        sys.modules.setdefault(name, m)
    base = importlib.util.spec_from_file_location("simglucose.patient.base", pkgdir + "/patient/base.py")
    bm = importlib.util.module_from_spec(base); sys.modules["simglucose.patient.base"] = bm; base.loader.exec_module(bm)
    s = importlib.util.spec_from_file_location("simglucose.patient.t1dpatient", pkgdir + "/patient/t1dpatient.py")
    m = importlib.util.module_from_spec(s); sys.modules["simglucose.patient.t1dpatient"] = m; s.loader.exec_module(m)
    _SG = m
    return m


@pytest.mark.parametrize("name", ["adolescent#001", "adult#005", "child#003"])
def test_matches_simglucose(name):
    sg = _load_simglucose_patient()
    ref = sg.T1DPatient(load_params([name]).iloc[0])
    ours = PatientBatch(load_params([name]))
    basal = ours.basal_u_per_min[0]
    ref_bg, our_bg = [], []
    for t in range(600):
        cho = 60.0 if t == 60 else 0.0
        ins = basal + (6.0 if t == 60 else 0.0)
        ref_bg.append(ref.observation.Gsub)
        our_bg.append(ours.bg[0])
        ref.step(sg.Action(CHO=cho, insulin=ins))
        ours.step(np.array([cho]), np.array([ins]))
    ref_bg, our_bg = np.array(ref_bg), np.array(our_bg)
    assert np.max(np.abs(ref_bg - our_bg)) < 0.5, np.max(np.abs(ref_bg - our_bg))
