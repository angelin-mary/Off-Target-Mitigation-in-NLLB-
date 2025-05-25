import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Replace this with your actual dataset
distance_df = pd.DataFrame({
    "Language Pair": [
    "arb-min", "arb-sat", "arb-taq", "arb-acm", "arb-ars",
    "arb-ary", "arb-aka", "arb-acq", "arb-prs", "arb-ajp",
    "arb-dyu", "arb-apc", "arb-aeb", "arb-arz", "arb-zho",
    "arb-hrv", "arb-ace", "arb-awa", "arb-bod", "arb-kin",
    "arb-bjn", "arb-knc"
],

"Distance": [
    0.587052, 0.624674, 0.240000, 0.488318, 0.650030,
    0.373901, 0.714226, 0.572785, 1.000000, 0.537640,
    0.641936, 0.368636, 1.000000, 0.382605, 0.605442,
    0.555125, 0.528595, 0.556160, 0.811438, 0.563436,
    0.594266, 0.756291
],

"Off-Target Rate": [
    100.00, 100.00, 100.00, 96.00, 96.00,
    89.00, 88.00, 85.00, 85.00, 80.00,
    80.00, 79.00, 72.00, 70.00, 43.00,
    42.00, 36.00, 36.00, 33.00, 23.00,
    21.00, 21.00
]

})
distance_df["Proximity"] = ["Close" if d <= 0.5 else "Distant" for d in distance_df["Distance"]]

pearson_corr, _ = pearsonr(distance_df["Distance"], distance_df["Off-Target Rate"])
spearman_corr, _ = spearmanr(distance_df["Distance"], distance_df["Off-Target Rate"])

plt.figure(figsize=(10, 6))
sns.regplot(x="Distance", y="Off-Target Rate", data=distance_df, scatter_kws={"s": 50}, line_kws={"color": "red"})
plt.title(f"Language Distance vs. Off-Target Rate\n(Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f})")
plt.xlabel("Language Distance (URIEL)")
plt.ylabel("Off-Target Rate (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Proximity", y="Off-Target Rate", data=distance_df)
plt.title("Off-Target Rate Distribution: Close vs. Distant Languages")
plt.xlabel("Language Proximity")
plt.ylabel("Off-Target Rate (%)")
plt.grid(True)
plt.show()
