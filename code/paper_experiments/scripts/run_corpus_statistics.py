"""Recompute core corpus statistics and figures from clean notebook cells.

Full-corpus NER is a separate, optional notebook pass; this script does not
replace the fresh test-set NER used for ASR evaluation.
"""

import json
import os
from pathlib import Path

from IPython.display import display

os.chdir(Path(__file__).resolve().parents[2])
os.environ["MPLBACKEND"] = "Agg"
n = json.loads(Path("code/statistics.ipynb").read_text())
ns = {"display": display, "__name__": "__main__"}
for i in [1, 3, 5, 7, 8, 10, 12, 13, 15, 16, 30, 33]:
    print("Basic corpus cell", i, flush=True)
    exec(compile("".join(n["cells"][i]["source"]), f"statistics.ipynb:cell{i}", "exec"), ns)
out = {}
for k in ["df", "cv"]:
    d = ns[k]
    out[k] = {
        "clips": len(d),
        "hours": float(d.duration.sum() / 3600),
        "mean_seconds": float(d.duration.mean()),
        "tokens": int(d.n_words.sum()),
    }
out["ney_matched_vocabulary"] = ns["ney_vocab_matched"]
out["cv_matched_vocabulary"] = ns["cv_vocab_matched"]
Path("results/paper_audit/basic_corpus_statistics.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
