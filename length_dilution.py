"""
Length-dilution experiments for the gender-name effect.

Hypothesis: on short template resumes the name is a large share of the text, so it can move the
embedding; on long real resumes it is diluted, so the gap vanishes.

A) TRUNCATE real resumes to L words (name always on top).
   Prediction if the hypothesis is true: the gap appears as L gets small.
B) PAD synthetic resumes with K words of real resume text appended.
   Prediction: the gap (seen at K=0) shrinks toward zero as K grows.
   K=0 is also a replication of the synthetic result with 64 new names.

Both use the same name-level test as before (unit = name, bootstrap CI over names), and the gap is also
expressed in score SDs so lengths are comparable.

Run:  python length_dilution.py --csv Resume.csv --models bge-base
      python length_dilution.py --csv Resume.csv --models bge-base MiniLM --n 60      (faster)
Needs biaslens.py and biaslens_real.py in the same folder.
Output: results_real/length_dilution.csv, results_real/length_dilution.png
"""
import argparse
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biaslens as S
import biaslens_real as B


def gender_gap(v, nrng):
    v = v.copy()
    v["adj"] = v["score"] - v.groupby("base_id")["score"].transform("mean")
    pn = v.groupby(["label", "gender"])["adj"].mean().reset_index()
    gap, lo, hi, p, _, _ = B.unit_level_gap(pn, "gender", "female", "male", nrng)
    sd = v.groupby("base_id")["score"].mean().std(ddof=1)
    return dict(gap=gap, ci_lo=lo, ci_hi=hi, p=p, gap_sd=gap / sd, ci_lo_sd=lo / sd, ci_hi_sd=hi / sd)


def score(model, key, df, jd_texts, jd_idx):
    X = B.embed(model, key, df["text"].tolist())
    J = B.embed(model, key, jd_texts)
    return (X * J[jd_idx]).sum(axis=1)


def run_truncate(base, model, key, lengths, assign, nrng):
    cats = list(B.JDS)
    jd_texts = [B.JDS[c] for c in cats]
    rows = []
    for L in lengths:
        recs = []
        for r in base.itertuples():
            body = " ".join(r.text.split()[:L])
            for g in B.NAME_GROUPS:
                nm = assign[(r.base_id, g)]
                recs.append((r.base_id, r.Category, g.split("_")[0], nm, f"{nm}\n{body}"))
        df = pd.DataFrame(recs, columns=["base_id", "category", "gender", "label", "text"])
        idx = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
        df["score"] = score(model, key, df, jd_texts, idx)
        res = gender_gap(df, nrng)
        res.update(model=key, exp="truncate_real", length=L)
        rows.append(res)
        print(f"  [truncate] L={L:4d}  gap={res['gap_sd']:+.3f} SD  p={res['p']:.3f}")
    return rows


def run_pad(specs, pad_src, model, key, pads, assign, nrng):
    roles = list(S.ROLES)
    jd_texts = [S.ROLES[r]["jd"] for r in roles]
    rows = []
    for K in pads:
        recs = []
        for s in specs:
            extra = " ".join(pad_src[s["id"]].split()[:K])
            for g in B.NAME_GROUPS:
                nm = assign[(s["id"], g)]
                text = S.render(s, nm, S.NEUTRAL_COLLEGE) + (("\n" + extra) if K else "")
                recs.append((s["id"], s["role"], g.split("_")[0], nm, text))
        df = pd.DataFrame(recs, columns=["base_id", "role", "gender", "label", "text"])
        idx = df["role"].map({r: i for i, r in enumerate(roles)}).to_numpy()
        df["score"] = score(model, key, df, jd_texts, idx)
        res = gender_gap(df, nrng)
        res.update(model=key, exp="pad_synthetic", length=K)
        rows.append(res)
        print(f"  [pad]      K={K:4d}  gap={res['gap_sd']:+.3f} SD  p={res['p']:.3f}")
    return rows


def plot(res):
    panels = [("truncate_real", "Real resumes truncated to L words", "words of real resume kept (L)"),
              ("pad_synthetic", "Synthetic resumes padded with real text", "words of real text appended (K)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (exp, title, xl) in zip(axes, panels):
        s = res[res.exp == exp]
        if s.empty:
            ax.set_visible(False)
            continue
        xs = sorted(s["length"].unique())
        models = list(dict.fromkeys(s["model"]))
        for i, m in enumerate(models):
            d = s[s.model == m].set_index("length").loc[xs]
            x = np.arange(len(xs)) + (i - (len(models) - 1) / 2) * 0.08
            ax.errorbar(x, d["gap_sd"], yerr=[d["gap_sd"] - d["ci_lo_sd"], d["ci_hi_sd"] - d["gap_sd"]],
                        marker="o", capsize=3, label=m)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_xlabel(xl)
        ax.set_ylabel("Female - male gap (score SDs, 95% CI over names)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(B.OUT / "length_dilution.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=120, help="real resumes per category (part A)")
    ap.add_argument("--n-synth", type=int, default=50, help="synthetic resumes per role (part B)")
    ap.add_argument("--models", nargs="+", default=["bge-base", "MiniLM"], choices=list(B.MODELS))
    ap.add_argument("--lengths", nargs="+", type=int, default=[20, 40, 80, 160, 320, 640])
    ap.add_argument("--pads", nargs="+", type=int, default=[0, 50, 100, 200, 400])
    ap.add_argument("--parts", nargs="+", default=["A", "B"], choices=["A", "B"])
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    B.OUT.mkdir(exist_ok=True)
    rng = random.Random(B.SEED)
    nrng = np.random.default_rng(B.SEED)

    base = B.load_resumes(args.csv, args.n)
    specs = S.make_base(args.n_synth, random.Random(S.SEED))
    avg_real = base["text"].str.split().str.len().mean()
    avg_syn = np.mean([len(S.render(s, "Candidate", S.NEUTRAL_COLLEGE).split()) for s in specs])
    print(f"Average length: real resumes {avg_real:.0f} words, synthetic resumes {avg_syn:.0f} words")
    print("(Models also truncate input: MiniLM at 256 tokens, bge/e5 at 512 tokens.)\n")

    # names fixed once, so only the length changes between conditions
    assign_real = {(r.base_id, g): rng.choice(n) for r in base.itertuples() for g, n in B.NAME_GROUPS.items()}
    assign_syn = {(s["id"], g): rng.choice(n) for s in specs for g, n in B.NAME_GROUPS.items()}
    pool = base.loc[base["Category"] == "INFORMATION-TECHNOLOGY", "text"].tolist()
    pad_src = {s["id"]: rng.choice(pool) for s in specs}

    rows = []
    for key in args.models:
        print(f"== {key}")
        model = SentenceTransformer(B.MODELS[key])
        if "A" in args.parts:
            rows += run_truncate(base, model, key, args.lengths, assign_real, nrng)
        if "B" in args.parts:
            rows += run_pad(specs, pad_src, model, key, args.pads, assign_syn, nrng)

    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for (m, e), g in res.groupby(["model", "exp"]):  # Holm across lengths within each curve
        res.loc[g.index, "p_holm"] = B.holm(g["p"])
    res.to_csv(B.OUT / "length_dilution.csv", index=False)
    plot(res)

    pd.set_option("display.width", 250)
    cols = ["model", "exp", "length", "gap_sd", "ci_lo_sd", "ci_hi_sd", "p", "p_holm"]
    print("\nRESULTS (gap in score SDs, female - male):")
    print(res[cols].round(4).to_string(index=False))
    print(f"\nSaved to {B.OUT.resolve()}")


if __name__ == "__main__":
    main()