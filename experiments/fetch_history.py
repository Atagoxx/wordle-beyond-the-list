"""Download past Wordle answers from the NYT endpoint the game itself uses:
https://www.nytimes.com/svc/wordle/v2/YYYY-MM-DD.json  ->  data/nyt_history.json (not committed; see README).

usage: python3 experiments/fetch_history.py [END]    END = YYYY-MM-DD or "yesterday" (default 2026-09-25,
the cutoff used for every number in EXPERIMENTS.md). Today's answer is never fetched, so nothing gets spoiled.
"""
import datetime
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATA

OUT = os.path.join(DATA, "nyt_history.json")


def get(d):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://www.nytimes.com/svc/wordle/v2/{d}.json", timeout=20) as r:
                j = json.load(r)
            time.sleep(0.1)
            return d, {"solution": j["solution"], "id": j.get("id"), "days_since_launch": j.get("days_since_launch"),
                       "editor": j.get("editor")}
        except Exception:
            time.sleep(2 ** attempt)
    return d, None


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "2026-09-25"
    today = datetime.date.today()
    end = today - datetime.timedelta(1) if arg == "yesterday" else datetime.date.fromisoformat(arg)
    end = min(end, today - datetime.timedelta(1))
    start = datetime.date(2021, 6, 19)
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    days = [(start + datetime.timedelta(n)).isoformat() for n in range((end - start).days + 1)]
    todo = [d for d in days if d not in data]
    t = time.time()
    with ThreadPoolExecutor(6) as ex:
        for d, v in ex.map(get, todo):
            if v:
                data[d] = v
    json.dump(data, open(OUT, "w"), indent=0, sort_keys=True)
    print(f"{len(data)}/{len(days)} days in {time.time()-t:.0f}s; missing: {[d for d in days if d not in data][:10]}")
