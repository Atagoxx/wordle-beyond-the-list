// Reads {words, weights, histories, pairs} as JSON on stdin; prints the engine's answers as JSON.
const E = require("./engine.js");
const inp = JSON.parse(require("fs").readFileSync(0, "utf8"));
E.init(inp.words, inp.weights);
const out = {
  rec: inp.histories.map(h => { const r = E.solve(h, { top: 1, likely: 1 }); return [r.count, r.rec[0] && r.rec[0].word]; }),
  codes: inp.pairs.map(([g, a]) => E.pattern(E.letters(g), E.indexOf(a))),
};
console.log(JSON.stringify(out));
