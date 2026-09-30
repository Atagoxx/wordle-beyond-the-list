// Wordle solver engine: no answer list, candidates weighted by a prior. Same scoring rule as the
// Python benchmark: expected-guesses cost from remaining entropy and the chance of winning outright.
var ENGINE = (function () {
  var N = 0, L = null, W = null;
  var cnt = new Int8Array(26), res = new Int8Array(5);

  // weights: either the packed base-36 prior string, or a ready array of per-word weights
  function init(wordStr, weights) {
    N = wordStr.length / 5;
    L = new Uint8Array(wordStr.length);
    for (var i = 0; i < wordStr.length; i++) L[i] = wordStr.charCodeAt(i) - 97;
    W = new Float64Array(N);
    for (var k = 0; k < N; k++) {
      W[k] = typeof weights === "string" ? Math.pow(10, -parseInt(weights.substr(k * 2, 2), 36) / 100) : weights[k];
    }
  }

  // feedback code of guess letters g (array of 5 ints) against word index a: sum(color_i * 3^i)
  function pattern(g, a) {
    var o = a * 5, i, code = 0;
    for (i = 0; i < 5; i++) {
      if (g[i] === L[o + i]) res[i] = 2;
      else { res[i] = 0; cnt[L[o + i]]++; }
    }
    for (i = 0; i < 5; i++) {
      if (res[i] === 0 && cnt[g[i]] > 0) { res[i] = 1; cnt[g[i]]--; }
    }
    for (i = 0; i < 5; i++) cnt[L[o + i]] = 0;
    return res[0] + 3 * res[1] + 9 * res[2] + 27 * res[3] + 81 * res[4];
  }

  function letters(word) {
    var g = [];
    for (var i = 0; i < 5; i++) g.push(word.charCodeAt(i) - 97);
    return g;
  }

  function wordAt(k) {
    var s = "";
    for (var i = 0; i < 5; i++) s += String.fromCharCode(97 + L[k * 5 + i]);
    return s;
  }

  function indexOf(word) {
    var lo = 0, hi = N - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1, w = wordAt(mid);
      if (w === word) return mid;
      if (w < word) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  }

  // history: [{word, code}] -> indices of words consistent with every row
  function candidates(history) {
    var S = [];
    var gs = history.map(function (h) { return letters(h.word); });
    for (var k = 0; k < N; k++) {
      var ok = true;
      for (var r = 0; r < history.length; r++) {
        if (pattern(gs[r], k) !== history[r].code) { ok = false; break; }
      }
      if (ok) S.push(k);
    }
    return S;
  }

  function rank(S, top, onProgress) {
    var n = S.length, total = 0, j;
    var p = new Float64Array(n);
    for (j = 0; j < n; j++) total += W[S[j]];
    var hs = 0;
    var pwinOf = new Float64Array(N);
    for (j = 0; j < n; j++) {
      p[j] = W[S[j]] / total;
      pwinOf[S[j]] = p[j];
      if (p[j] > 0) hs -= p[j] * Math.log2(p[j]);
    }
    if (n <= 2) {
      var best = 0;
      for (j = 1; j < n; j++) if (p[j] > p[best]) best = j;
      return { hs: hs, list: [{ k: S[best], cost: 1 + (1 - p[best]), h: 0, pwin: p[best] }] };
    }
    var mass = new Float64Array(243), g = [0, 0, 0, 0, 0];
    var bestList = [];
    for (var k = 0; k < N; k++) {
      var o = k * 5;
      for (var i = 0; i < 5; i++) g[i] = L[o + i];
      mass.fill(0);
      for (j = 0; j < n; j++) mass[pattern(g, S[j])] += p[j];
      var h = 0;
      for (var c = 0; c < 243; c++) { var m = mass[c]; if (m > 0) h -= m * Math.log2(m); }
      var rem = Math.max(hs - h, 0), pw = pwinOf[k];
      var cost = 1 + (1 - pw) * (1 + 0.55 * rem + 0.1 * rem * rem);
      if (bestList.length < top || cost < bestList[bestList.length - 1].cost) {
        bestList.push({ k: k, cost: cost, h: h, pwin: pw });
        bestList.sort(function (a, b) { return a.cost - b.cost || a.k - b.k; });
        if (bestList.length > top) bestList.pop();
      }
      if (onProgress && (k & 1023) === 0) onProgress(k / N);
    }
    return { hs: hs, list: bestList };
  }

  function solve(history, opts) {
    opts = opts || {};
    var S = candidates(history);
    var out = { count: S.length, rec: [], likely: [] };
    if (!S.length) return out;
    var total = 0, j;
    for (j = 0; j < S.length; j++) total += W[S[j]];
    var byProb = S.slice().sort(function (a, b) { return W[b] - W[a] || a - b; });
    out.likely = byProb.slice(0, opts.likely || 60).map(function (k) {
      return { word: wordAt(k), p: W[k] / total };
    });
    var r = opts.noRank ? { hs: 0, list: [] } : rank(S, opts.top || 5, opts.onProgress);
    if (opts.noRank) for (j = 0; j < S.length; j++) { var pj = W[S[j]] / total; if (pj > 0) r.hs -= pj * Math.log2(pj); }
    out.entropy = r.hs;
    out.rec = r.list.map(function (x) {
      return { word: wordAt(x.k), cost: x.cost, bits: x.h, pwin: x.pwin };
    });
    if (opts.forced) {
      // a fixed move (the opener) goes first; any one-step picks become alternatives
      var g = letters(opts.forced), mass = new Float64Array(243), pw = 0, h = 0;
      for (j = 0; j < S.length; j++) {
        var q = W[S[j]] / total;
        mass[pattern(g, S[j])] += q;
        if (wordAt(S[j]) === opts.forced) pw = q;
      }
      for (var c = 0; c < 243; c++) if (mass[c] > 0) h -= mass[c] * Math.log2(mass[c]);
      out.rec = [{ word: opts.forced, cost: 0, bits: h, pwin: pw }].concat(
        out.rec.filter(function (x) { return x.word !== opts.forced; }).slice(0, 4));
    }
    return out;
  }

  return { init: init, solve: solve, indexOf: indexOf, pattern: pattern, letters: letters, wordAt: wordAt,
           size: function () { return N; } };
})();
if (typeof module !== "undefined") module.exports = ENGINE;
