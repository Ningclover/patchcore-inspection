"""
Generate all-black dummy ground truth masks for PatchCore evaluation.

For each image in <dataset>/test/bad/, creates a matching black PNG mask
in <dataset>/ground_truth/bad/. Used when no pixel-level annotations exist.

Usage:
    python generate_dummy_masks.py --dataset /path/to/my_data/LarASIC
    python generate_dummy_masks.py --dataset /path/to/my_data/LarASIC --anomaly_type bad
"""

import argparse
import os

from PIL import Image


def generate_masks(dataset_path, anomaly_type="bad"):
    bad_dir = os.path.join(dataset_path, "test", anomaly_type)
    mask_dir = os.path.join(dataset_path, "ground_truth", anomaly_type)

    if not os.path.isdir(bad_dir):
        raise FileNotFoundError(f"Test folder not found: {bad_dir}")

    os.makedirs(mask_dir, exist_ok=True)

    # Remove stale masks
    for f in os.listdir(mask_dir):
        os.remove(os.path.join(mask_dir, f))

    images = sorted(os.listdir(bad_dir))
    for fname in images:
        img = Image.open(os.path.join(bad_dir, fname))
        mask = Image.new("L", img.size, 0)
        mask_name = os.path.splitext(fname)[0] + "_mask.png"
        mask.save(os.path.join(mask_dir, mask_name))

    print(f"Created {len(images)} masks in {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset folder (e.g. .../my_data/LarASIC)")
    parser.add_argument("--anomaly_type", default="bad", help="Subfolder name under test/ (default: bad)")
    args = parser.parse_args()

    generate_masks(args.dataset, args.anomaly_type)
