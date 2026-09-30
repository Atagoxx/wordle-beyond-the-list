"""Assemble the single-file web tool: tool/template.html + engine.js + packed prior + answer history.

usage: python3 tool/build.py      -> dist/wordle-solver.html
Needs data/nyt_history.json (experiments/fetch_history.py). The built page embeds that history, so
dist/ is not committed; see the README before publishing a built page.
"""
import datetime
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

G = open(os.path.join(DATA, "allowed.txt")).read().split()
N = len(G)
A = set(open(os.path.join(DATA, "answers.txt")).read().split())
bits = "".join("1" if w in A else "0" for w in G) + "000"
orig_hex = "".join(format(int(bits[i:i + 4], 2), "x") for i in range(0, N, 4))
assert len(orig_hex) == (N + 3) // 4

H = json.load(open(os.path.join(DATA, "nyt_history.json")))
days = sorted(H)
start = datetime.date.fromisoformat(days[0])
for i, d in enumerate(days):  # the page indexes the history by day number, so no gaps allowed
    assert datetime.date.fromisoformat(d) == start + datetime.timedelta(i), d
hist = "".join(H[d]["solution"] for d in days)
assert all(H[d]["solution"] in set(G) for d in days)

# walk-forward backtest numbers shown on the page (results/backtest_summary.json from experiments/backtest.py)
S = json.load(open(os.path.join(ROOT, "results", "backtest_summary.json")))["periods"]
allp = {v: S[v]["all"] for v in S}
openers = {"salet": {"avg": round(allp["P2s"]["avg_found"], 3), "fails": allp["P2s"]["over6"], "le3": round(allp["P2s"]["le3"], 3)},
           "tares": {"avg": round(allp["P2"]["avg_found"], 3), "fails": allp["P2"]["over6"], "le3": round(allp["P2"]["le3"], 3)}}
sim = [
    ("本工具：历史感知先验 + SALET 开局", f"{allp['P2s']['avg_found']:.2f} 次", f"{allp['P2s']['not_found']} 天找不到答案"),
    ("同上，改用 TARES 开局", f"{allp['P2']['avg_found']:.2f} 次", f"{allp['P2']['not_found']} 天找不到答案"),
    ("上一版本工具（不看历史答案）", f"{allp['P0']['avg_found']:.2f} 次", f"{allp['P0']['not_found']} 天找不到答案"),
    ("教科书最优：原始名单 + SALET", f"{allp['T0']['avg_found']:.2f} 次 *", f"{allp['T0']['not_found']} 天找不到答案"),
    ("教科书 + 去掉已经出过的词", f"{allp['T1']['avg_found']:.2f} 次 *", f"{allp['T1']['not_found']} 天找不到答案"),
]
simrows = "".join(f'<tr><td>{a}</td><td class="n">{b}</td><td class="n">{c}</td></tr>' for a, b, c in sim)

d = json.load(open(os.path.join(DATA, "tool_prior.json")))
html = open(os.path.join(ROOT, "tool", "template.html")).read()
html = (html.replace("{{ENGINE}}", open(os.path.join(ROOT, "tool", "engine.js")).read())
            .replace("{{WORDS}}", d["words"]).replace("{{WEIGHTS}}", d["weights"]).replace("{{ORIG}}", orig_hex)
            .replace("{{HIST_START}}", days[0]).replace("{{HIST}}", hist)
            .replace("{{OPENERS}}", json.dumps(openers)).replace("{{SIMROWS}}", simrows))
assert "{{" not in html
# template.html is a page fragment (the claude.ai artifact host adds the document shell); add one for standalone use
html = ('<!doctype html>\n<html lang="zh-CN">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + html + "\n</html>\n")
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
open(os.path.join(ROOT, "dist", "wordle-solver.html"), "w").write(html)
print(f"wrote dist/wordle-solver.html ({len(html) // 1024} KB); history {days[0]} .. {days[-1]} ({len(days)} days)")
