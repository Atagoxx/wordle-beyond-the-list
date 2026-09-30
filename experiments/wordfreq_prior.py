"""The free baseline prior: wordfreq's Zipf frequency for every accepted guess (pip install wordfreq)."""
import json
import os

from wordfreq import zipf_frequency

from common import G, PRIORS

if __name__ == "__main__":
    json.dump({w: zipf_frequency(w, "en") for w in G}, open(os.path.join(PRIORS, "wordfreq_zipf.json"), "w"))
    print("wrote data/priors/wordfreq_zipf.json")
