import pandas as pd
from scipy.stats import mannwhitneyu
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)

v = pd.read_csv("results/variants.csv"); s = pd.read_csv("results/scores.csv")
v["name"] = v["text"].str.split("\n").str[0]
s["name"] = list(v["name"]) * s["model"].nunique()
n = s[(s.exp == "name") & (s.group != "neutral")].copy()
n["gender"] = n["group"].str.split("_").str[0]

# remove resume-to-resume variation: score relative to that resume's mean across name variants
n["adj"] = n["score"] - n.groupby(["model", "base_id"])["score"].transform("mean")
pn = n.groupby(["model", "name", "gender"])["adj"].mean().reset_index()
pn.to_csv("results/per_name.csv", index=False)

print("\nNAME-LEVEL TEST (20 female vs 20 male names)")
for m, g in pn.groupby("model"):
    f = g[g.gender == "female"]["adj"]; ma = g[g.gender == "male"]["adj"]
    p = mannwhitneyu(f, ma).pvalue
    print(f"{m:10s} female {f.mean():+.4f}  male {ma.mean():+.4f}  p = {p:.4f}")

print("\nSHORTLIST RATE (fair = 0.20)")
rows = []
for (m, r), g in n.groupby(["model", "role"]):
    top = g.nlargest(int(len(g) * 0.2), "score").index
    rows.append(g.assign(sel=g.index.isin(top)))
print(pd.concat(rows).groupby(["group", "model"])["sel"].mean().unstack().round(3))

print("\nPER-NAME ADJUSTED SCORE")
print(pn.pivot(index="name", columns="model", values="adj").round(4).sort_values("bge-base"))