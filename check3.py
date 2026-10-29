import pandas as pd
from scipy.stats import mannwhitneyu
v = pd.read_csv("results/variants.csv"); s = pd.read_csv("results/scores.csv")
s["college"] = list(v["text"].str.extract(r"Computer Science, (.*?), \d{4}-")[0]) * s["model"].nunique()
c = s[s.exp == "college"].copy()
c["adj"] = c["score"] - c.groupby(["model", "base_id"])["score"].transform("mean")
pc = c.groupby(["model", "group", "college"])["adj"].mean().reset_index()
for m, g in pc.groupby("model"):
    a = g[g.group == "tier1"]["adj"]; b = g[g.group == "tier3"]["adj"]
    print(f"{m:10s} tier1 {a.mean():+.4f}  tier3 {b.mean():+.4f}  p = {mannwhitneyu(a, b).pvalue:.4f}")