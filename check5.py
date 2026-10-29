import pandas as pd
pd.set_option("display.width", 250)
a = pd.read_csv("results_real/length_dilution_bge-base.csv")
b = pd.read_csv("results_real/length_dilution_MiniLM-e5.csv")
r = pd.concat([a, b])
print(r[["model", "exp", "length", "gap", "ci_lo", "ci_hi", "p"]].round(4).to_string(index=False))