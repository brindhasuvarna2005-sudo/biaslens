import pandas as pd
from scipy.stats import mannwhitneyu
u = pd.read_csv("results_real/per_unit.csv")
c = u[u.exp == "college"].copy()
c["len"] = c["label"].str.len()
print("MEAN ADJUSTED SCORE BY TIER")
print(c.pivot_table(index="model", columns="group", values="adj").round(4))
print("\nTIER CONTRASTS (college-level Mann-Whitney)")
for m, d in c.groupby("model"):
    t = {g: x["adj"].to_numpy() for g, x in d.groupby("group")}
    for a, b in [("tier1", "tier3"), ("tier2", "tier3"), ("tier1", "tier2")]:
        print(f"{m:10s} {a}-{b}: {t[a].mean()-t[b].mean():+.4f}  p={mannwhitneyu(t[a], t[b]).pvalue:.3f}")
print("\nAVG NAME LENGTH BY TIER")
print(c.groupby("group")["len"].mean().round(1))
for m, d in c.groupby("model"):
    print(f"{m:10s} corr(length, adj) = {d['len'].corr(d['adj']):+.2f}")