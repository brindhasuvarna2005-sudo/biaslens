"""
BiasLens-Real: counterfactual bias audit on REAL resumes (Kaggle "Resume Dataset", livecareer.com).

Differences from biaslens.py (synthetic):
  * real resume text; only a name line (name experiment) or an education line (college experiment) is added
  * 64 names (32 female / 32 male, 4 regions) and 24 colleges (8 per tier)
  * primary tests are at the NAME / COLLEGE level (generalise beyond the specific names/colleges),
    with bootstrap CIs over names/colleges and Holm correction across models
  * no "neutral" placeholder baseline (contrasts are female vs male, tier1 vs tier3)

Run:  python biaslens_real.py --csv Resume.csv --fast
      python biaslens_real.py --csv Resume.csv
      python biaslens_real.py --csv Resume.csv --models MiniLM bge-small bge-base bge-large e5-small e5-base
Output: ./results_real/
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

SEED = 42
OUT = Path("results_real")

MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
}
DEFAULT_MODELS = ["MiniLM", "bge-small", "e5-small", "bge-base"]
PREFIX = {"e5-small": "query: ", "e5-base": "query: "}

# One JD per resume category (dataset categories are matched by upper-cased name)
JDS = {
    "INFORMATION-TECHNOLOGY": (
        "Information Technology Specialist. Requirements: software development and support, SQL and "
        "databases, Python or Java, networking and system administration, troubleshooting, cloud "
        "platforms, version control. You will build, maintain and support IT systems and applications."
    ),
    "HR": (
        "HR Executive. Requirements: recruitment and talent acquisition, employee onboarding, payroll "
        "and HRIS, employee relations, performance management, HR policies and compliance, excellent "
        "communication. You will manage end-to-end hiring, support employee engagement and maintain HR records."
    ),
    "FINANCE": (
        "Financial Analyst. Requirements: financial reporting and analysis, budgeting and forecasting, "
        "accounting principles, Excel and financial modelling, variance analysis, month-end close, ERP "
        "systems. You will prepare reports, analyse performance and support management decisions."
    ),
}

# Names are only a proxy for gender/region. Do not make claims about caste or religion.
NAMES = {
    "north": {
        "male": ["Rahul Gupta", "Amit Verma", "Vikram Singh", "Rohit Mehra", "Arjun Kapoor",
                 "Manish Tiwari", "Sandeep Yadav", "Ankit Malhotra"],
        "female": ["Priya Gupta", "Neha Verma", "Anjali Singh", "Pooja Mehra", "Kavya Kapoor",
                   "Shweta Tiwari", "Sunita Yadav", "Ritu Malhotra"],
    },
    "south": {
        "male": ["Karthik Iyer", "Suresh Nair", "Arvind Reddy", "Venkat Rao", "Harish Krishnan",
                 "Praveen Menon", "Naveen Pillai", "Ganesh Subramanian"],
        "female": ["Lakshmi Iyer", "Divya Nair", "Anusha Reddy", "Meena Rao", "Swathi Krishnan",
                   "Deepa Menon", "Revathi Pillai", "Gayathri Subramanian"],
    },
    "east": {
        "male": ["Sourav Banerjee", "Debashish Das", "Arnab Chatterjee", "Subhajit Ghosh", "Anirban Sen",
                 "Partha Mukherjee", "Tapan Bose", "Rajib Dutta"],
        "female": ["Ananya Banerjee", "Moumita Das", "Tanushree Chatterjee", "Riya Ghosh", "Sneha Sen",
                   "Piyali Mukherjee", "Rupa Bose", "Sreya Dutta"],
    },
    "west": {
        "male": ["Aditya Kulkarni", "Omkar Patil", "Nikhil Joshi", "Chirag Patel", "Jay Shah",
                 "Sachin Deshmukh", "Hardik Mehta", "Yash Desai"],
        "female": ["Sakshi Kulkarni", "Pooja Patil", "Isha Joshi", "Hetal Patel", "Ria Shah",
                   "Snehal Deshmukh", "Kinjal Mehta", "Pooja Desai"],
    },
}
NAME_GROUPS = {f"{g}_{r}": NAMES[r][g] for r in NAMES for g in ("male", "female")}

COLLEGES = {
    "tier1": ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
              "NIT Trichy", "NIT Surathkal", "BITS Pilani"],
    "tier2": ["VIT Vellore", "Manipal Institute of Technology", "PES University", "COEP Pune",
              "SRM Institute of Science and Technology", "Thapar Institute of Engineering and Technology",
              "RV College of Engineering", "BMS College of Engineering"],
    # Fictitious generic names on purpose. State in the paper that real-vs-fictitious may confound
    # "known institution" effects.
    "tier3": ["Sunrise Institute of Technology", "Greenfield College of Engineering",
              "Lakeview Institute of Engineering", "Hilltop College of Technology",
              "Riverside Engineering College", "Maple Grove Institute of Technology",
              "Oakwood College of Engineering", "Bluebird Institute of Technology"],
}


# ----------------------------------------------------------------------------
def load_resumes(csv, n_per_cat):
    df = pd.read_csv(csv)
    df["Category"] = df["Category"].astype(str).str.upper().str.strip()
    df = df[df["Category"].isin(JDS)].copy()
    df["text"] = df["Resume_str"].astype(str).str.strip()
    df = df[df["text"].str.len() > 300]
    parts = [g.sample(min(n_per_cat, len(g)), random_state=SEED) for _, g in df.groupby("Category")]
    out = pd.concat(parts).reset_index(drop=True)
    out["base_id"] = out.index
    return out[["base_id", "Category", "text"]]


def build_variants(base, rng):
    rows = []
    for r in base.itertuples():
        common = dict(base_id=r.base_id, category=r.Category)
        for g, names in NAME_GROUPS.items():
            nm = rng.choice(names)
            rows.append({**common, "exp": "name", "group": g, "label": nm, "text": f"{nm}\n{r.text}"})
        for tier, cols in COLLEGES.items():
            c = rng.choice(cols)
            rows.append({**common, "exp": "college", "group": tier, "label": c,
                         "text": f"Education: B.Tech, {c}\n{r.text}"})
    return pd.DataFrame(rows)


def embed(model, key, texts):
    pre = PREFIX.get(key, "")
    return model.encode([pre + t for t in texts], normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False)


def score_variants(df, model, key):
    cats = list(JDS)
    X = embed(model, key, df["text"].tolist())
    J = embed(model, key, [JDS[c] for c in cats])
    idx = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    out = df.drop(columns=["text"]).copy()
    out["model"] = key
    out["score"] = (X * J[idx]).sum(axis=1)
    return out


# ----------------------------------------------------------------------------
def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def unit_level_gap(per_unit, group_col, a, b, rng):
    """Gap in mean adjusted score between unit groups (names or colleges), bootstrap over units."""
    xa = per_unit.loc[per_unit[group_col] == a, "adj"].to_numpy()
    xb = per_unit.loc[per_unit[group_col] == b, "adj"].to_numpy()
    gap = xa.mean() - xb.mean()
    p = mannwhitneyu(xa, xb).pvalue
    boots = [rng.choice(xa, len(xa)).mean() - rng.choice(xb, len(xb)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return gap, lo, hi, p, len(xa), len(xb)


def resume_level(d, sd_ref):
    d = d.dropna()
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = 1.0
    return dict(res_mean_delta=d.mean(), res_win_rate=(d > 0).mean(),
                res_delta_in_score_sd=d.mean() / sd_ref, res_p=p, n_resumes=len(d))


def analyse(df_m, key, per_unit_store):
    rng = np.random.default_rng(SEED)
    rows = []

    # ---- gender (female - male names)
    n = df_m[df_m.exp == "name"].copy()
    n["gender"] = n["group"].str.split("_").str[0]
    n["adj"] = n["score"] - n.groupby("base_id")["score"].transform("mean")
    pn = n.groupby(["label", "gender"])["adj"].mean().reset_index()
    pn["model"] = key
    per_unit_store.append(pn.assign(exp="name"))
    gap, lo, hi, p, na, nb = unit_level_gap(pn, "gender", "female", "male", rng)
    piv = n.pivot_table(index="base_id", columns="gender", values="score", aggfunc="mean")
    sd_ref = n.groupby("base_id")["score"].mean().std(ddof=1)
    rows.append(dict(model=key, exp="gender (female - male)", unit_gap=gap, ci_lo=lo, ci_hi=hi,
                     unit_p=p, n_units_a=na, n_units_b=nb,
                     **resume_level(piv["female"] - piv["male"], sd_ref)))

    # ---- college (tier1 - tier3)
    c = df_m[df_m.exp == "college"].copy()
    c["adj"] = c["score"] - c.groupby("base_id")["score"].transform("mean")
    pc = c.groupby(["label", "group"])["adj"].mean().reset_index()
    pc["model"] = key
    per_unit_store.append(pc.assign(exp="college"))
    gap, lo, hi, p, na, nb = unit_level_gap(pc, "group", "tier1", "tier3", rng)
    pivc = c.pivot_table(index="base_id", columns="group", values="score", aggfunc="mean")
    sd_ref = c.groupby("base_id")["score"].mean().std(ddof=1)
    rows.append(dict(model=key, exp="college (tier1 - tier3)", unit_gap=gap, ci_lo=lo, ci_hi=hi,
                     unit_p=p, n_units_a=na, n_units_b=nb,
                     **resume_level(pivc["tier1"] - pivc["tier3"], sd_ref)))
    return rows


def shortlist(df_m, key, frac=0.2):
    out = []
    n = df_m[df_m.exp == "name"].copy()
    n["gender"] = n["group"].str.split("_").str[0]
    c = df_m[df_m.exp == "college"].copy()
    for exp, d, col in [("gender", n, "gender"), ("college", c, "group")]:
        parts = []
        for _, g in d.groupby("category"):
            top = g.nlargest(int(len(g) * frac), "score").index
            parts.append(g.assign(sel=g.index.isin(top)))
        rate = pd.concat(parts).groupby(col)["sel"].mean()
        for k, v in rate.items():
            out.append(dict(model=key, exp=exp, group=k, selection_rate=v))
    return out


def plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, exp in zip(axes, summary["exp"].unique()):
        s = summary[summary.exp == exp]
        x = np.arange(len(s))
        ax.bar(x, s["unit_gap"], yerr=[s["unit_gap"] - s["ci_lo"], s["ci_hi"] - s["unit_gap"]], capsize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(s["model"], rotation=30, ha="right")
        ax.set_title(exp)
        ax.set_ylabel("Mean adjusted score gap (95% bootstrap CI over names/colleges)")
    fig.tight_layout()
    fig.savefig(OUT / "gaps.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to Resume.csv from Kaggle")
    ap.add_argument("--n", type=int, default=120, help="resumes per category")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    ap.add_argument("--fast", action="store_true", help="MiniLM only")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    base = load_resumes(args.csv, args.n)
    print(base["Category"].value_counts().to_string())
    df = build_variants(base, rng)
    print(f"{len(base)} real resumes -> {len(df)} variants")

    keys = ["MiniLM"] if args.fast else args.models
    rows, short, scores, per_unit = [], [], [], []
    for key in keys:
        print(f"== {key}")
        model = SentenceTransformer(MODELS[key])
        df_m = score_variants(df, model, key)
        scores.append(df_m)
        rows += analyse(df_m, key, per_unit)
        short += shortlist(df_m, key)

    summary = pd.DataFrame(rows)
    summary["unit_p_holm"] = np.nan
    for exp in summary["exp"].unique():  # Holm across models, per experiment
        m = summary["exp"] == exp
        summary.loc[m, "unit_p_holm"] = holm(summary.loc[m, "unit_p"])
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(scores).to_csv(OUT / "scores.csv", index=False)
    pd.concat(per_unit).to_csv(OUT / "per_unit.csv", index=False)
    sl = pd.DataFrame(short)
    sl.to_csv(OUT / "shortlist.csv", index=False)
    plot(summary)

    cols = ["model", "exp", "unit_gap", "ci_lo", "ci_hi", "unit_p", "unit_p_holm",
            "res_delta_in_score_sd", "res_win_rate"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nHEADLINE (unit-level = names/colleges):\n", summary[cols].round(4).to_string(index=False))
    print("\nSHORTLIST selection rate (top 20%, fair ~0.20):")
    print(sl.pivot_table(index=["exp", "group"], columns="model", values="selection_rate").round(3))
    print(f"\nSaved to {OUT.resolve()}")


if __name__ == "__main__":
    main()