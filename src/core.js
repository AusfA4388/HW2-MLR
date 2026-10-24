/* ===== MLR core: pure functions, no DOM (tested in node against statsmodels) ===== */
const MLR = (() => {
  // ---------- PRNG / utils ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function shuffled(n, seed) { const r = mulberry32(seed), idx = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return idx; }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };

  // ---------- CSV ----------
  function parseCSV(text) {
    const rows = []; let row = [], f = '', q = false;
    text = text.replace(/^﻿/, '');
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ',') { row.push(f); f = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; }
      else f += c;
    }
    if (f !== '' || row.length) { row.push(f); rows.push(row); }
    const header = rows.shift().map(h => h.trim().replace(/\s+/g, ' '));
    return { header, rows: rows.filter(r => r.length === header.length) };
  }
  const isMissing = v => v === undefined || /^\s*(|na|nan|null|none|\?)\s*$/i.test(v);
  const isNum = v => /^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$/.test(v);

  // describe columns: numeric / categorical / dropped(reason)
  function describeColumns(csv) {
    return csv.header.map((name, j) => {
      const vals = csv.rows.map(r => r[j]); const present = vals.filter(v => !isMissing(v));
      const missing = vals.length - present.length;
      const numeric = present.length > 0 && present.every(isNum);
      const levels = [...new Set(present.map(v => v.trim()))].sort();
      let kind = numeric ? 'numeric' : 'categorical', reason = '';
      if (/^(id|.*[_ ]id|index|unnamed.*)$/i.test(name)) { kind = 'dropped'; reason = 'ID 欄位'; }
      else if (!numeric && levels.length > 15) { kind = 'dropped'; reason = `類別過多（${levels.length} 種）`; }
      else if (levels.length <= 1) { kind = 'dropped'; reason = '常數欄位'; }
      return { name, j, kind, reason, missing, levels };
    });
  }

  // build design matrix (dummy coding, drop_first) + listwise deletion
  function buildDesign(csv, cols, target, logTarget) {
    const tcol = cols.find(c => c.name === target);
    const used = cols.filter(c => c.kind !== 'dropped' && c.name !== target);
    const featNames = [], featSrc = [];
    used.forEach(c => { if (c.kind === 'numeric') { featNames.push(c.name); featSrc.push({ col: c.name, kind: 'num' }); }
      else c.levels.slice(1).forEach(l => { featNames.push(`${c.name}_${l}`); featSrc.push({ col: c.name, kind: 'cat', level: l, base: c.levels[0] }); }); });
    const X = [], y = [], rowsKept = []; let dropped = 0, nonPos = 0;
    csv.rows.forEach((r, i) => {
      if (isMissing(r[tcol.j]) || used.some(c => isMissing(r[c.j]))) { dropped++; return; }
      let yv = +r[tcol.j]; if (logTarget) { if (!(yv > 0)) { nonPos++; return; } }
      const x = []; used.forEach(c => { if (c.kind === 'numeric') x.push(+r[c.j]); else { const v = r[c.j].trim(); c.levels.slice(1).forEach(l => x.push(v === l ? 1 : 0)); } });
      X.push(x); y.push(yv); rowsKept.push(i);
    });
    // drop dummy/numeric columns that are constant after deletion
    const keep = featNames.map((_, k) => { const v = X.map(x => x[k]); return Math.max(...v) !== Math.min(...v); });
    return { X: X.map(x => x.filter((_, k) => keep[k])), y, featNames: featNames.filter((_, k) => keep[k]), featSrc: featSrc.filter((_, k) => keep[k]), used, dropped, nonPos, n0: csv.rows.length };
  }

  // ---------- linear algebra ----------
  function cholSolveInv(A, b, wantInv) { // SPD solve with tiny jitter fallback
    const n = A.length; let jitter = 0;
    for (let attempt = 0; attempt < 6; attempt++) {
      const L = Array.from({ length: n }, () => new Float64Array(n)); let ok = true;
      for (let i = 0; i < n && ok; i++) for (let j = 0; j <= i; j++) {
        let s = A[i][j] + (i === j ? jitter * (A[i][i] || 1) : 0);
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) { if (s <= 0) { ok = false; break; } L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j];
      }
      if (!ok) { jitter = jitter ? jitter * 100 : 1e-10; continue; }
      const fwd = v => { const z = new Float64Array(n); for (let i = 0; i < n; i++) { let s = v[i]; for (let k = 0; k < i; k++) s -= L[i][k] * z[k]; z[i] = s / L[i][i]; } return z; };
      const bwd = z => { const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; };
      const sol = b ? Array.from(bwd(fwd(b))) : null; let inv = null;
      if (wantInv) { inv = []; for (let c = 0; c < n; c++) { const e = new Float64Array(n); e[c] = 1; inv.push(Array.from(bwd(fwd(e)))); } }
      return { sol, inv, jitter };
    }
    throw new Error('矩陣奇異，無法求解（可能有完全共線的欄位）');
  }
  // Gram of [1, X] over a set of rows
  function gram(X, y, rows) {
    const p = X[0].length + 1, G = Array.from({ length: p }, () => new Float64Array(p)), Xy = new Float64Array(p); let yy = 0;
    for (const i of rows) { const z = [1, ...X[i]]; for (let a = 0; a < p; a++) { Xy[a] += z[a] * y[i]; for (let b = a; b < p; b++) G[a][b] += z[a] * z[b]; } yy += y[i] * y[i]; }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    return { G, Xy, yy, n: rows.length };
  }
  const sub = (g, idx) => ({ G: idx.map(a => idx.map(b => g.G[a][b])), Xy: idx.map(a => g.Xy[a]) });
  // SSE on gram set gv for coefficients b on columns idx
  function sseOn(gv, idx, b) { let s = gv.yy; for (let a = 0; a < idx.length; a++) { s -= 2 * b[a] * gv.Xy[idx[a]]; for (let c = 0; c < idx.length; c++) s += b[a] * b[c] * gv.G[idx[a]][idx[c]]; } return Math.max(s, 0); }

  // ---------- distributions ----------
  function lgamma(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const v of c) s += v / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  function betacf(a, b, x) { let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-16) break; } return h; }
  function ibeta(x, a, b) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
  const tTwoSidedP = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);
  const fUpperP = (F, d1, d2) => F <= 0 ? 1 : ibeta(d2 / (d2 + d1 * F), d2 / 2, d1 / 2);
  function tQuantile975(df) { let lo = 0, hi = 50; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (tTwoSidedP(m, df) > 0.05) lo = m; else hi = m; } return (lo + hi) / 2; }
  function normInv(p) { const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const q = Math.min(p, 1 - p); let x; if (q > 0.02425) { const u = p - 0.5, r = u * u; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    const r = Math.sqrt(-2 * Math.log(q)); x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1); return p < 0.5 ? x : -x; }

  // ---------- CV machinery (gram-based, O(k^3) per subset) ----------
  function makeCV(X, y, trainRows, K, seed) {
    const order = shuffled(trainRows.length, seed).map(i => trainRows[i]);
    const folds = [...Array(K)].map((_, f) => order.filter((_, i) => i % K === f));
    return folds.map((val, f) => ({ tr: gram(X, y, order.filter((_, i) => i % K !== f)), va: gram(X, y, val) }));
  }
  function cvScore(cv, feats) { // feats are 0-based feature indices; returns {mean, se} of fold RMSE
    const idx = [0, ...feats.map(j => j + 1)];
    const r = cv.map(({ tr, va }) => { const s = sub(tr, idx); const b = cholSolveInv(s.G, s.Xy).sol; return Math.sqrt(sseOn(va, idx, b) / va.n); });
    return { mean: mean(r), se: sd(r) / Math.sqrt(r.length) };
  }

  // ---------- feature rankings ----------
  function standardize(X, rows) { const p = X[0].length, mu = [], s = []; for (let j = 0; j < p; j++) { const v = rows.map(i => X[i][j]); mu.push(mean(v)); s.push(sd(v) || 1); } return { mu, s, Z: rows.map(i => X[i].map((v, j) => (v - mu[j]) / s[j])) }; }
  function rankForward(cv, p) { const sel = [], rem = [...Array(p).keys()]; while (rem.length) { let best = -1, bs = Infinity; for (const c of rem) { const sc = cvScore(cv, [...sel, c]).mean; if (sc < bs) { bs = sc; best = c; } } sel.push(best); rem.splice(rem.indexOf(best), 1); } return sel; }
  function rankRFE(X, y, rows) { // recursive feature elimination by |standardized coefficient|
    const { Z } = standardize(X, rows), yv = rows.map(i => y[i]), ZZ = Z, Y = yv; let alive = [...Array(X[0].length).keys()]; const elim = [];
    const g = gram(ZZ, Y, [...ZZ.keys()]);
    while (alive.length > 1) { const idx = [0, ...alive.map(j => j + 1)], s = sub(g, idx), b = cholSolveInv(s.G, s.Xy).sol; let w = 0; for (let a = 1; a < idx.length; a++) if (Math.abs(b[a]) < Math.abs(b[w + 1])) w = a - 1; elim.push(alive[w]); alive.splice(w, 1); }
    return [...alive, ...elim.reverse()];
  }
  function rankPath(X, y, rows, l1ratio) { // order in which features enter the lasso / elastic-net path
    const { Z } = standardize(X, rows), yv = rows.map(i => y[i]), ym = mean(yv), r0 = yv.map(v => v - ym), n = Z.length, p = Z[0].length;
    const corr = [...Array(p).keys()].map(j => Math.abs(Z.reduce((s, z, i) => s + z[j] * r0[i], 0)) / n);
    const lmax = Math.max(...corr) / l1ratio, beta = new Float64Array(p), r = r0.slice(), entry = new Array(p).fill(Infinity);
    for (let s = 0; s < 100; s++) { const lam = lmax * Math.pow(1e-4, s / 99);
      for (let it = 0; it < 200; it++) { let mx = 0;
        for (let j = 0; j < p; j++) { let rho = 0; for (let i = 0; i < n; i++) rho += Z[i][j] * r[i]; rho = rho / n + beta[j];
          const nb = Math.sign(rho) * Math.max(Math.abs(rho) - lam * l1ratio, 0) / (1 + lam * (1 - l1ratio)), d = nb - beta[j];
          if (d) { for (let i = 0; i < n; i++) r[i] -= d * Z[i][j]; beta[j] = nb; mx = Math.max(mx, Math.abs(d)); } }
        if (mx < 1e-7) break; }
      for (let j = 0; j < p; j++) if (beta[j] !== 0 && entry[j] === Infinity) entry[j] = s; }
    return [...Array(p).keys()].sort((a, b) => entry[a] - entry[b] || corr[b] - corr[a]);
  }

  // ---------- OLS with full inference ----------
  function ols(X, y, rows, feats) {
    const idx = [0, ...feats.map(j => j + 1)], g = gram(X, y, rows), s = sub(g, idx), { sol: b, inv } = cholSolveInv(s.G, s.Xy, true);
    const n = rows.length, k = feats.length, df = n - k - 1, sse = sseOn(g, idx, b), sigma2 = sse / df, ybar = mean(rows.map(i => y[i]));
    const sst = rows.reduce((t, i) => t + (y[i] - ybar) ** 2, 0), r2 = 1 - sse / sst, adj = 1 - (1 - r2) * (n - 1) / df, tq = tQuantile975(df);
    const se = b.map((_, a) => Math.sqrt(sigma2 * inv[a][a]));
    const coef = b.map((v, a) => ({ name: a ? null : '(Intercept)', b: v, se: se[a], t: v / se[a], p: tTwoSidedP(v / se[a], df), lo: v - tq * se[a], hi: v + tq * se[a] }));
    const F = ((sst - sse) / k) / sigma2;
    // VIF = diag of inverse correlation matrix of selected features
    let vif = feats.map(() => 1);
    if (k > 1) { const { Z } = standardize(feats.map ? rows.map(i => feats.map(j => X[i][j])) : [], [...rows.keys()]); const C = gram(Z, Z.map(() => 0), [...Z.keys()]).G.slice(1).map(r => Array.from(r).slice(1).map(v => v / (Z.length - 1)));
      vif = cholSolveInv(C, null, true).inv.map((r, a) => r[a]); }
    const predict = x => { // x: full feature vector; returns fit, mean-CI and prediction-interval half widths
      const z = [1, ...feats.map(j => x[j])]; let f = 0, q = 0; for (let a = 0; a < z.length; a++) { f += b[a] * z[a]; for (let c = 0; c < z.length; c++) q += z[a] * inv[a][c] * z[c]; }
      return { fit: f, ci: tq * Math.sqrt(sigma2 * q), pi: tq * Math.sqrt(sigma2 * (1 + q)) }; };
    return { feats, b, coef, vif, n, k, df, sigma: Math.sqrt(sigma2), r2, adj, F, Fp: fUpperP(F, k, df), tq, predict };
  }

  function metrics(yt, yp, k) { const n = yt.length, m = mean(yt), sse = yt.reduce((s, v, i) => s + (v - yp[i]) ** 2, 0), sst = yt.reduce((s, v) => s + (v - m) ** 2, 0), r2 = 1 - sse / sst;
    return { rmse: Math.sqrt(sse / n), mae: yt.reduce((s, v, i) => s + Math.abs(v - yp[i]), 0) / n, r2, adj: n - k - 1 > 0 ? 1 - (1 - r2) * (n - 1) / (n - k - 1) : NaN }; }

  // ---------- whole pipeline ----------
  const METHODS = [
    { id: 'forward', name: 'Forward SFS (CV)' },
    { id: 'rfe', name: 'RFE（標準化係數）' },
    { id: 'lasso', name: 'Lasso path' },
    { id: 'enet', name: 'ElasticNet path (α=0.5)' }];

  function run(D, opts = {}) {
    const seed = opts.seed ?? 42, K = 5, n = D.y.length, p = D.featNames.length;
    if (n < 20) throw new Error(`有效樣本只有 ${n} 筆，太少無法分析`);
    const tf = opts.logTarget ? v => Math.log(v) : v => v, inv = opts.logTarget ? v => Math.exp(v) : v => v;
    const ty = D.y.map(tf);
    const perm = shuffled(n, seed), nTest = Math.round(n * 0.2), test = perm.slice(0, nTest).sort((a, b) => a - b), train = perm.slice(nTest).sort((a, b) => a - b);
    const maxK = Math.min(p, train.length - 10);
    const cv = makeCV(D.X, ty, train, K, seed + 1);
    const rankings = {
      forward: rankForward(cv, p).slice(0, maxK), rfe: rankRFE(D.X, ty, train), lasso: rankPath(D.X, ty, train, 1), enet: rankPath(D.X, ty, train, 0.5) };
    const curves = {}, picks = {};
    for (const m of METHODS) {
      const order = rankings[m.id]; const c = [];
      for (let k = 1; k <= maxK; k++) { const f = order.slice(0, k), s = cvScore(cv, f), mod = ols(D.X, ty, train, f);
        const yp = test.map(i => inv(mod.predict(D.X[i]).fit)); const mt = metrics(test.map(i => D.y[i]), yp, k);
        c.push({ k, cv: s.mean, se: s.se, testRmse: mt.rmse, testR2: mt.r2 }); }
      const best = c.reduce((a, b) => b.cv < a.cv ? b : a), one = c.find(r => r.cv <= best.cv + best.se);
      curves[m.id] = c; picks[m.id] = { min: best.k, oneSE: one.k, row: one };
    }
    const bestMethod = METHODS.map(m => m.id).reduce((a, b) => picks[b].row.cv < picks[a].row.cv ? b : a);
    const kSel = picks[bestMethod].oneSE, sel = rankings[bestMethod].slice(0, kSel);
    const model = ols(D.X, ty, train, sel), full = ols(D.X, ty, train, [...Array(p).keys()].slice(0, maxK));
    const evalModel = (mod) => { const pr = idx => idx.map(i => inv(mod.predict(D.X[i]).fit));
      return { train: metrics(train.map(i => D.y[i]), pr(train), mod.k), test: metrics(test.map(i => D.y[i]), pr(test), mod.k) }; };
    const testPred = test.map(i => { const q = model.predict(D.X[i]); return { i, y: D.y[i], fit: inv(q.fit), ciLo: inv(q.fit - q.ci), ciHi: inv(q.fit + q.ci), piLo: inv(q.fit - q.pi), piHi: inv(q.fit + q.pi) }; });
    const coverage = testPred.filter(r => r.y >= r.piLo && r.y <= r.piHi).length / testPred.length;
    const trainResid = train.map(i => { const f = model.predict(D.X[i]).fit; return { fit: f, res: ty[i] - f }; });
    model.coef.forEach((c, a) => { if (a) c.name = D.featNames[sel[a - 1]]; });
    const trainMeans = [...Array(p).keys()].map(j => mean(train.map(i => D.X[i][j])));
    return { n, p, train, test, rankings, curves, picks, bestMethod, kSel, sel, model, full, evalSel: evalModel(model), evalFull: evalModel(full), testPred, coverage, trainResid, trainMeans, inv, log: !!opts.logTarget };
  }
  return { parseCSV, describeColumns, buildDesign, run, ols, METHODS, normInv, mean, sd, tQuantile975, tTwoSidedP };
})();
if (typeof module !== 'undefined') module.exports = MLR;
