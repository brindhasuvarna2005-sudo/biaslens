"""
College-effect control: is it 'real vs fictitious institution', 'name length', or prestige?

Adds a 4th group of FICTITIOUS but SHORT names (~same length as the IITs) next to your existing groups:
  tier1_real_short   (IIT Delhi, ...            ~11 chars, real)
  tier2_real_long    (VIT Vellore, ...          ~25 chars, real)
  tier3_fict_long    (Greenfield College of ... ~31 chars, fictitious)
  fict_short         (Apex Tech, ...            ~10 chars, fictitious)

Contrasts:
  tier1_real_short - fict_short      -> real vs fictitious, length matched
  tier3_fict_long  - fict_short      -> length effect, realness matched (both fictitious)
  tier2_real_long  - tier3_fict_long -> real vs fictitious, roughly length matched

Run: python college_control.py --csv Resume.csv                  (MiniLM + bge-base)
     python college_control.py --csv Resume.csv --models MiniLM bge-small e5-small bge-base
Output: ./results_real/college_control.csv
"""
import argparse
import random

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import biaslens_real as B

FICT_SHORT = ["Apex Tech", "Zenith Tech", "Vertex Tech", "Nova Tech",
              "Orion Tech", "Crest Tech", "Summit Tech", "Pinnacle Tech"]

CONTRASTS = [
    ("tier1_real_short", "fict_short", "REAL vs fictitious (both short)"),
    ("tier3_fict_long", "fict_short", "LONG vs short (both fictitious)"),
    ("tier2_real_long", "tier3_fict_long", "REAL vs fictitious (both long)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--models", nargs="+", default=["MiniLM", "bge-base"], choices=list(B.MODELS))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    rng = random.Random(B.SEED)
    base = B.load_resumes(args.csv, args.n)
    groups = {
        "tier1_real_short": B.COLLEGES["tier1"],
        "tier2_real_long": B.COLLEGES["tier2"],
        "tier3_fict_long": B.COLLEGES["tier3"],
        "fict_short": FICT_SHORT,
    }
    rows = []
    for r in base.itertuples():
        for g, cols in groups.items():
            c = rng.choice(cols)
            rows.append(dict(base_id=r.base_id, category=r.Category, group=g, label=c,
                             text=f"Education: B.Tech, {c}\n{r.text}"))
    df = pd.DataFrame(rows)
    df["len"] = df["label"].str.len()
    print("Avg college-name length per group:")
    print(df.groupby("group")["len"].mean().round(1).to_string(), "\n")

    cats = list(B.JDS)
    idx = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    out = []
    for key in args.models:
        print(f"== {key}")
        model = SentenceTransformer(B.MODELS[key])
        X = B.embed(model, key, df["text"].tolist())
        J = B.embed(model, key, [B.JDS[c] for c in cats])
        d = df.drop(columns=["text"]).copy()
        d["score"] = (X * J[idx]).sum(axis=1)
        d["adj"] = d["score"] - d.groupby("base_id")["score"].transform("mean")
        pc = d.groupby(["label", "group"])["adj"].mean().reset_index()
        t = {g: x["adj"].to_numpy() for g, x in pc.groupby("group")}
        print("Mean adjusted score by group:", {g: round(float(v.mean()), 4) for g, v in t.items()})
        for a, b, desc in CONTRASTS:
            gap = t[a].mean() - t[b].mean()
            p = mannwhitneyu(t[a], t[b]).pvalue
            print(f"  {desc:36s} {a} - {b}: {gap:+.4f}  p={p:.3f}")
            out.append(dict(model=key, contrast=desc, a=a, b=b, gap=gap, p=p))

    pd.DataFrame(out).to_csv(B.OUT / "college_control.csv", index=False)
    print(f"\nSaved to {B.OUT / 'college_control.csv'}")


if __name__ == "__main__":
    main()