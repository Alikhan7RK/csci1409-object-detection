"""Dataset report: split sizes, objects per class, label sanity checks, near-duplicate check,
and ground-truth visualisations.

Outputs (in results/data/<dataset-name>/):
    stats.md / stats.json      images and objects per split and class
    label_issues.csv           every suspicious label found
    duplicates.csv             near-duplicate image pairs (perceptual hash)
    gt_samples.jpg             grid of training images with ground-truth boxes
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
COLORS = ["tab:orange", "tab:blue", "tab:red", "tab:green", "tab:purple"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(path):
    rows = []
    if not path.exists():
        return None
    for ln, line in enumerate(path.read_text().strip().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        rows.append((ln, parts))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "african-wildlife.yaml"))
    ap.add_argument("--out", default=None, help="default: results/data/<dataset yaml name>")
    ap.add_argument("--phash-threshold", type=int, default=10,
                    help="max pHash Hamming distance for a near-duplicate candidate")
    ap.add_argument("--dhash-threshold", type=int, default=10,
                    help="max dHash Hamming distance to confirm the candidate (both must hold)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    names = cfg["names"]
    nc = len(names)
    ds = Path(cfg["path"])
    out = Path(args.out or ROOT / "results" / "data" / Path(args.data).stem)
    out.mkdir(parents=True, exist_ok=True)

    stats = {}
    issues = []
    hashes = []  # (split, filename, hash)
    box_areas = defaultdict(list)

    for split in SPLITS:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        obj = Counter()
        img_with = Counter()
        empty = 0
        for img in imgs:
            lp = lbl_dir / (img.stem + ".txt")
            rows = read_labels(lp)
            try:
                with Image.open(img) as im:
                    w, h = im.size
                    rgb = im.convert("RGB")
                    hashes.append((split, img.name, imagehash.phash(rgb), imagehash.dhash(rgb)))
            except Exception as e:  # corrupt image
                issues.append([split, img.name, "", "corrupt_image", str(e)])
                continue
            if rows is None:
                issues.append([split, img.name, "", "missing_label_file", ""])
                continue
            if not rows:
                empty += 1
                issues.append([split, img.name, "", "empty_label_file (no objects)", ""])
                continue
            seen = set()
            classes_here = set()
            for ln, parts in rows:
                if len(parts) != 5:
                    issues.append([split, img.name, ln, "wrong_number_of_fields", " ".join(parts)])
                    continue
                c, x, y, bw, bh = int(float(parts[0])), *map(float, parts[1:])
                if not 0 <= c < nc:
                    issues.append([split, img.name, ln, "invalid_class_id", c])
                    continue
                if not all(0 <= v <= 1 for v in (x, y, bw, bh)):
                    issues.append([split, img.name, ln, "coords_outside_0_1", " ".join(parts)])
                if x - bw / 2 < -0.01 or x + bw / 2 > 1.01 or y - bh / 2 < -0.01 or y + bh / 2 > 1.01:
                    issues.append([split, img.name, ln, "box_extends_outside_image", " ".join(parts)])
                if bw * w < 4 or bh * h < 4:
                    issues.append([split, img.name, ln, "tiny_box (<4 px side)", f"{bw*w:.1f}x{bh*h:.1f}px"])
                key = (c, round(x, 3), round(y, 3), round(bw, 3), round(bh, 3))
                if key in seen:
                    issues.append([split, img.name, ln, "duplicate_box", " ".join(parts)])
                seen.add(key)
                obj[names[c]] += 1
                classes_here.add(names[c])
                box_areas[names[c]].append(bw * bh)
            for c in classes_here:
                img_with[c] += 1
        stats[split] = {
            "images": len(imgs),
            "images_without_objects": empty,
            "objects_total": sum(obj.values()),
            "objects_per_class": {names[i]: obj[names[i]] for i in range(nc)},
            "images_containing_class": {names[i]: img_with[names[i]] for i in range(nc)},
        }

    # ---- near-duplicate check (pHash) ----
    # A pair is a near-duplicate when BOTH perceptual hashes are close. pHash alone gave
    # false positives (different animals on similar savanna backgrounds) at distance 10-12.
    def label_set(split, name):
        rows = read_labels(ds / "labels" / split / (Path(name).stem + ".txt")) or []
        return sorted(names[int(float(r[1][0]))] for r in rows)

    dups = []
    for i in range(len(hashes)):
        si, ni, pi, di = hashes[i]
        for j in range(i + 1, len(hashes)):
            sj, nj, pj, dj = hashes[j]
            dp = pi - pj
            if dp <= args.phash_threshold and (di - dj) <= args.dhash_threshold:
                la, lb = label_set(si, ni), label_set(sj, nj)
                dups.append([si, ni, sj, nj, dp, di - dj,
                             "CROSS-SPLIT" if si != sj else "same split",
                             "same" if la == lb else f"DIFFERENT: {la} vs {lb}"])
    cross = [d for d in dups if d[6] == "CROSS-SPLIT"]
    for d in dups:  # same image labelled differently = labelling inconsistency
        if d[7] != "same":
            issues.append([d[0], d[1], "", "duplicate_image_with_different_labels",
                           f"{d[2]}/{d[3]}: {d[7]}"])

    with open(out / "label_issues.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "line", "issue", "detail"])
        wr.writerows(issues)
    with open(out / "duplicates.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split_a", "image_a", "split_b", "image_b", "phash_distance", "dhash_distance",
                     "type", "class_labels"])
        wr.writerows(sorted(dups, key=lambda r: r[4]))

    summary = {
        "splits": stats,
        "label_issue_counts": dict(Counter(i[3] for i in issues)),
        "near_duplicate_pairs": len(dups),
        "cross_split_near_duplicate_pairs": len(cross),
        "phash_threshold": args.phash_threshold,
        "dhash_threshold": args.dhash_threshold,
        "median_box_area_fraction": {k: sorted(v)[len(v) // 2] for k, v in box_areas.items()},
    }
    json.dump(summary, open(out / "stats.json", "w"), indent=2)

    # ---- markdown table ----
    cls = [names[i] for i in range(nc)]
    md = ["| Split | Images | " + " | ".join(cls) + " | Total objects |",
          "|---|---|" + "---|" * len(cls) + "---|"]
    for s in SPLITS:
        st = stats[s]
        md.append(f"| {s} | {st['images']} | " +
                  " | ".join(str(st["objects_per_class"][c]) for c in cls) +
                  f" | {st['objects_total']} |")
    tot_img = sum(stats[s]["images"] for s in SPLITS)
    tot = {c: sum(stats[s]["objects_per_class"][c] for s in SPLITS) for c in cls}
    md.append(f"| **total** | **{tot_img}** | " + " | ".join(f"**{tot[c]}**" for c in cls) +
              f" | **{sum(tot.values())}** |")
    (out / "stats.md").write_text("\n".join(md) + "\n")

    # ---- ground-truth grid ----
    rnd = random.Random(args.seed)
    train_imgs = sorted((ds / "images" / "train").iterdir())
    # pick at least one image per class, then fill to 8
    picks, need = [], set(cls)
    for p in rnd.sample(train_imgs, len(train_imgs)):
        rows = read_labels(ds / "labels" / "train" / (p.stem + ".txt")) or []
        here = {names[int(float(r[1][0]))] for r in rows}
        if here & need or (len(picks) < 8 and not need):
            picks.append(p)
            need -= here
        if len(picks) == 8:
            break
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, p in zip(axes.flat, picks):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        ax.imshow(im)
        for _, parts in read_labels(ds / "labels" / "train" / (p.stem + ".txt")) or []:
            c, x, y, bw, bh = int(float(parts[0])), *map(float, parts[1:])
            ax.add_patch(patches.Rectangle(((x - bw / 2) * w, (y - bh / 2) * h), bw * w, bh * h,
                                           fill=False, lw=2.5, ec=COLORS[c % len(COLORS)]))
            ax.text((x - bw / 2) * w, (y - bh / 2) * h, names[c], color="white", fontsize=11,
                    bbox=dict(fc=COLORS[c % len(COLORS)], alpha=0.85, pad=1, lw=0))
        ax.set_title(p.name, fontsize=10)
        ax.axis("off")
    plt.suptitle("Ground-truth boxes (training split)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out / "gt_samples.jpg", dpi=80)

    print("\n".join(md))
    print(json.dumps({k: v for k, v in summary.items() if k != "splits"}, indent=2))
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
