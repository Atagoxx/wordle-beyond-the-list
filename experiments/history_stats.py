"""What the real answer history looks like: per year, how many answers came from outside the original
2,315-word list, and how many repeated an earlier answer. Needs data/nyt_history.json (fetch_history.py)."""
import collections
import json
import os

from common import ANS, DATA, INDEX, save_json

if __name__ == "__main__":
    H = json.load(open(os.path.join(DATA, "nyt_history.json")))
    days = sorted(H)
    seen, rows, repeats = {}, collections.OrderedDict(), []
    for d in days:
        w = H[d]["solution"]
        y = d[:4]
        r = rows.setdefault(y, {"days": 0, "outside_original": 0, "repeat": 0, "not_accepted_guess": 0,
                                "outside_examples": []})
        r["days"] += 1
        if w not in ANS:
            r["outside_original"] += 1
            if len(r["outside_examples"]) < 8:
                r["outside_examples"].append(w)
        if w not in INDEX:
            r["not_accepted_guess"] += 1
        if w in seen:
            r["repeat"] += 1
            repeats.append({"date": d, "word": w, "first_used": seen[w]})
        else:
            seen[w] = d
    unused = sorted(w for w in ANS if w not in seen)
    print(f"{days[0]} .. {days[-1]}: {len(days)} days, {len(seen)} distinct answers")
    print(f"{'year':<6}{'days':>6}{'outside original list':>24}{'repeats':>10}   examples outside the list")
    for y, r in rows.items():
        print(f"{y:<6}{r['days']:>6}{r['outside_original']:>24}{r['repeat']:>10}   {', '.join(r['outside_examples'])}")
    print(f"first repeat: {repeats[0] if repeats else None}; repeats so far: {len(repeats)}")
    print(f"original-list words never used yet: {len(unused)}")
    save_json("history_stats.json", {"range": [days[0], days[-1]], "per_year": rows, "repeats": repeats,
                                     "unused_original_count": len(unused)})
