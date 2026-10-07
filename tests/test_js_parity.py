"""The browser twin (web/twin.js) must match the Python twin."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dosetwin.twin import model as M

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    th = M.default_theta(); th[0] = np.log(2.3); th[5] = 4.0; th[6] = 0.2
    z0 = M.steady_state(th, 0.01, 160.0); z0[9] = 0.4; z0[4] = 20.0
    H = 300
    u = np.zeros(H); u[0] = 4.5
    c = np.zeros(H); c[0] = 60.0
    steps = np.zeros(H); steps[30:60] = 100.0
    py = M.forecast(th, z0, u, 0.01, c, steps, 55.0, 1200, H)
    js = f"""
    const T = require('{ROOT / 'web' / 'twin.js'}');
    const steps = new Array({H}).fill(0); for (let k=30;k<60;k++) steps[k]=100;
    const r = T.forecast({json.dumps(th.tolist())}, {json.dumps(z0.tolist())},
      {{b:0.01, W:55.0, minute0:1200, H:{H}, bolus:4.5, carbs:60.0, steps}});
    console.log(JSON.stringify(Array.from(r)));
    """
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    assert np.max(np.abs(np.array(json.loads(out)) - py)) < 1e-6
