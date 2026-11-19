"""Build a leak-free copy of the dataset.

The official African Wildlife split contains the same photo in train AND val/test
(found by scripts/data_report.py). Rule used here (deterministic, no randomness):

    1. Group images with union-find over the near-duplicate pairs in duplicates.csv.
    2. If a group spans more than one split, move every member of the group to TRAIN.
       (Train gets priority so that val/test contain only photos the model never saw.)
    3. All other images keep their official split.

Output: datasets/african-wildlife-clean/ (images + labels) and
        datasets/african-wildlife-clean.yaml, plus results/data/split_changes.csv
"""
import argparse
import csv
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "african-wildlife.yaml"))
    ap.add_argument("--dups", default=str(ROOT / "results" / "data" / "african-wildlife" / "duplicates.csv"))
    ap.add_argument("--name", default="african-wildlife-clean")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    src = Path(cfg["path"])
    dst = ROOT / "datasets" / args.name

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in csv.DictReader(open(args.dups)):
        union((r["split_a"], r["image_a"]), (r["split_b"], r["image_b"]))

    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)

    new_split = {}  # (split, name) -> split
    for members in groups.values():
        if len({s for s, _ in members}) > 1:
            for m in members:
                new_split[m] = "train"

    if dst.exists():
        shutil.rmtree(dst)
    changes = []
    counts = {s: 0 for s in SPLITS}
    for split in SPLITS:
        for img in sorted((src / "images" / split).iterdir()):
            target = new_split.get((split, img.name), split)
            # prefix with the original split so names stay unique after moving
            new_name = img.name if target == split else f"from-{split}_{img.name}"
            (dst / "images" / target).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / target).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / target / new_name)
            lbl = src / "labels" / split / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst / "labels" / target / (Path(new_name).stem + ".txt"))
            counts[target] += 1
            if target != split:
                changes.append([img.name, split, target, new_name])

    cfg_out = dict(cfg, path=str(dst.resolve()))
    yaml.safe_dump(cfg_out, open(ROOT / "datasets" / f"{args.name}.yaml", "w"), sort_keys=False)

    out_csv = ROOT / "results" / "data" / "split_changes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "original_split", "new_split", "new_filename"])
        wr.writerows(changes)

    print(f"Moved {len(changes)} images to train to remove cross-split duplicates")
    print("New split sizes:", counts)
    print("Wrote", ROOT / "datasets" / f"{args.name}.yaml")


if __name__ == "__main__":
    main()
