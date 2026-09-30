"""Fix the first guess, then follow the tool's one-step rule with the tool's base prior (no answer list).
Plays all 2,315 original answers per opener.

usage: python3 experiments/openers.py tares,salet,crane,...
"""
import sys
import time
from multiprocessing import Pool

from common import evaluate, save_json, tool_prior

W = tool_prior()


def run(first):
    avg, fails, dist, _ = evaluate(W, first=first)
    le3 = sum(c for k, c in dist.items() if k <= 3) / sum(dist.values())
    return {"opener": first, "avg": avg, "le3": le3, "over6": fails, "worst": max(dist), "dist": dist}


if __name__ == "__main__":
    words = (sys.argv[1] if len(sys.argv) > 1 else
             "tares,salet,crane,soare,slate,trace,raise,roate,crate,reast,rates,lares").split(",")
    t = time.time()
    with Pool(4) as p:
        res = sorted(p.map(run, words), key=lambda r: r["avg"])
    save_json("openers.json", res)
    for r in res:
        print(f"{r['opener'].upper():6} avg {r['avg']:.4f}  solved in <=3: {r['le3']*100:.1f}%  over 6: {r['over6']}  "
              f"worst {r['worst']}  {r['dist']}")
    print(f"{time.time()-t:.0f}s")
