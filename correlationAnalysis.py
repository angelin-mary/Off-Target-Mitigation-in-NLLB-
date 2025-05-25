import pandas as pd
import itertools
import matplotlib.pyplot as plt
from lang2vec.lang2vec import distance

target_languages = [
    "arb_Latn", "min_Arab", "sat_Olck", "taq_Tfng", "acm_Arab", "ars_Arab", "ary_Arab",
    "aka_Latn", "acq_Arab", "prs_Arab", "ajp_Arab", "dyu_Latn", "apc_Arab", "aeb_Arab",
    "arz_Arab", "kmb_Latn", "zho_Hant", "hrv_Latn", "ace_Arab", "awa_Deva", "bod_Tibt",
    "kin_Latn", "arb_Arab", "bjn_Arab", "knc_Arab"
]

# Convert target languages to ISO 639-3 codes
iso_targets = [lang.split("_")[0] for lang in target_languages]

# Compute pairwise genetic distances
pairs = list(itertools.combinations(iso_targets, 2)) 
distances = [
    {"language_pair": f"{pair[0]}-{pair[1]}", "distance": distance("genetic", pair[0], pair[1])}
    for pair in pairs
]

pairwise_df = pd.DataFrame(distances)

threshold = 0.5  # Adjust as needed

pairwise_df["proximity"] = pairwise_df["distance"].apply(lambda x: "Close" if x <= threshold else "Distant")

print("\nPairwise Language Distances and Proximity Classification:")
print(pairwise_df)

close_pairs = pairwise_df[pairwise_df["proximity"] == "Close"]
distant_pairs = pairwise_df[pairwise_df["proximity"] == "Distant"]

print(f"\nNumber of Close Pairs: {len(close_pairs)}")
print(f"Number of Distant Pairs: {len(distant_pairs)}")

distance_matrix = pd.DataFrame(index=iso_targets, columns=iso_targets, dtype=float)
for index, row in pairwise_df.iterrows():
    lang1, lang2 = row["language_pair"].split("-")
    distance_matrix.loc[lang1, lang2] = row["distance"]
    distance_matrix.loc[lang2, lang1] = row["distance"] 

for lang in iso_targets:
    distance_matrix.loc[lang, lang] = 0

plt.figure(figsize=(12, 10))
plt.imshow(distance_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Genetic Distance")
plt.xticks(range(len(iso_targets)), iso_targets, rotation=90)
plt.yticks(range(len(iso_targets)), iso_targets)
plt.title("Genetic Distance Heatmap")
plt.tight_layout()
plt.show()
