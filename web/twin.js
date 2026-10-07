// DoseTwin forward model - exact port of dosetwin/twin/model.py (step/forecast).
// Parity with Python is checked by tests/test_js_parity.py.
const DoseTwinModel = (() => {
  const KE = 0.138, P2 = 0.02, TS = 8.0, TE = 60.0, TD = 30.0, VG = 1.6;
  const activity = (s) => Math.min(1, Math.max(0, (s - 40) / 80));
  const dawnShape = (m) => { const x = ((m % 1440) - 360) / 60; return Math.exp(-0.5 * x * x); };
  function step(z, th, u, b, c, steps, sd, W, minute, decayD) {
    const SI = Math.exp(th[0] - (th[8] || 0) * sd), Gb = th[1], fc = Math.exp(th[2]), tm = Math.exp(th[3]);
    const ti = Math.exp(th[4]), alpha = th[5], dawn = th[6], SG = Math.exp(th[7]);
    const [S1, S2, I, X, D1, D2, E, G, Gs, d] = z;
    const Ib = b / KE, Ri = S2 / ti, Ra = fc * D2 / tm * 1000 / (VG * W), a = activity(steps);
    const dG = -SG * (G - Gb) - X * G - alpha * E * G * 1e-3 + Ra + dawn * dawnShape(minute) + d;
    z[0] = S1 + u + b - S1 / ti;
    z[1] = S2 + (S1 - S2) / ti;
    z[2] = I + Ri - KE * I;
    z[3] = X + (-P2 * X + P2 * SI * (I - Ib) / W);
    z[4] = D1 + c - D1 / tm;
    z[5] = D2 + (D1 - D2) / tm;
    z[6] = E + (a - E) / TE;
    z[7] = Math.max(G + dG, 10);
    z[8] = Gs + (G - Gs) / TS;
    if (decayD) z[9] = d - d / TD;
  }
  // future inputs: arrays (or null) of length >= H
  function forecast(th, z0, opts) {
    const { b, W, minute0, H, bolus = 0, carbs = 0, steps = null, sleepDebt = 0 } = opts;
    const z = z0.slice(), out = new Float64Array(H + 1);
    out[0] = z[8];
    for (let k = 0; k < H; k++) {
      step(z, th, k === 0 ? bolus : 0, b, k === 0 ? carbs : 0, steps ? steps[k] : 0, sleepDebt, W, minute0 + k, true);
      out[k + 1] = z[8];
    }
    return out;
  }
  function risk(g) { const f = 1.509 * (Math.pow(Math.log(Math.min(600, Math.max(20, g))), 1.084) - 5.381); return 10 * f * f; }
  return { step, forecast, risk };
})();
if (typeof module !== "undefined") module.exports = DoseTwinModel;
