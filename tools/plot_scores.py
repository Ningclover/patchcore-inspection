"""Plot anomaly score distribution, colored by good/bad label."""
import csv
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/nfs/data/1/xning/elec_AI/results/evaluated_results/LarASIC_infer_images/mvtec_LarASIC_per_image_scores.csv"
)

good_scores = []
bad_scores = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        score = float(row["anomaly_score"])
        if "/good/" in row["image_path"]:
            good_scores.append(score)
        else:
            bad_scores.append(score)

good_scores = np.array(good_scores)
bad_scores = np.array(bad_scores)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

bins = np.linspace(0, 1, 41)

axes[0].hist(good_scores, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].set_title(f"Good images (n={len(good_scores)})", fontsize=13)
axes[0].set_ylabel("Count")
axes[0].axvline(good_scores.mean(), color="navy", linestyle="--", label=f"mean={good_scores.mean():.3f}")
axes[0].axvline(np.median(good_scores), color="cyan", linestyle=":", label=f"median={np.median(good_scores):.3f}")
axes[0].legend()

axes[1].hist(bad_scores, bins=bins, color="tomato", edgecolor="white", alpha=0.85)
axes[1].set_title(f"Bad images (n={len(bad_scores)})", fontsize=13)
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Anomaly score (min-max normalized)")
axes[1].axvline(bad_scores.mean(), color="darkred", linestyle="--", label=f"mean={bad_scores.mean():.3f}")
axes[1].axvline(np.median(bad_scores), color="salmon", linestyle=":", label=f"median={np.median(bad_scores):.3f}")
axes[1].legend()

fig.suptitle("PatchCore anomaly score distribution\n" + os.path.basename(CSV_PATH), fontsize=14)
plt.tight_layout()

out_path = CSV_PATH.replace(".csv", "_score_dist.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# Also print summary stats
print(f"\nGood  — mean={good_scores.mean():.4f}  std={good_scores.std():.4f}  min={good_scores.min():.4f}  max={good_scores.max():.4f}")
print(f"Bad   — mean={bad_scores.mean():.4f}  std={bad_scores.std():.4f}  min={bad_scores.min():.4f}  max={bad_scores.max():.4f}")
