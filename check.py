import pandas as pd
v = pd.read_csv("results/variants.csv"); s = pd.read_csv("results/scores.csv")
v["name"] = v["text"].str.split("\n").str[0]
s["name"] = list(v["name"]) * s["model"].nunique()
n = s[(s.exp == "name") & (s.group != "neutral")]

rows = []
for (m, r), g in n.groupby(["model", "role"]):
    top = g.nlargest(int(len(g) * 0.2), "score").index
    rows.append(g.assign(sel=g.index.isin(top)))
print("\nSHORTLIST RATE (fair = 0.20)")
print(pd.concat(rows).groupby(["group", "model"])["sel"].mean().unstack().round(3))
print("\nMEAN SCORE PER NAME")
print(n.groupby(["name", "model"])["score"].mean().unstack().round(4).sort_values("bge-base"))

sm = pd.read_csv("results/summary.csv")
t = sm[sm.contrast.isin(["female_avg - male_avg", "tier1 - tier3"])]
t = t[["model", "contrast", "mean_delta", "delta_in_score_sd", "win_rate", "p_adj"]].round(4)
t.to_csv("results/headline_table.csv", index=False)
print("\nHEADLINE TABLE"); print(t.to_string(index=False))