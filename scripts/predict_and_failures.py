"""Run the final model on the TEST split, save prediction examples and find failure cases.

Every prediction (conf >= --conf, after NMS) is matched greedily, highest confidence first, to
the ground truth:
  TP               same class, IoU >= 0.5
  wrong_class      overlaps a GT box of another class with IoU >= 0.5
  poor_box         same class but 0.1 <= IoU < 0.5 (object found, box inaccurate)
  false_positive   anything else (detection on background)
  missed (FN)      GT box with no matched prediction
Outputs (results/test_predictions/):
  predictions_grid.jpg       6 random test images with predicted boxes, class and confidence
  failures_<type>.jpg        worst examples of each failure type (left: GT, right: prediction)
  per_image.csv              counts per image, used to pick failure cases
  summary.json               totals and precision/recall at this operating point
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["tab:orange", "tab:blue", "tab:red", "tab:green"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def load_gt(lbl, w, h):
    gt = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
                gt.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return gt


def match(preds, gt):
    """preds: list of (cls, conf, box). Returns list of pred outcomes and list of missed GT idx."""
    used = set()
    outcomes = []
    for c, cf, b in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0, None
        for j, (gc, gb) in enumerate(gt):
            if j in used or gc != c:
                continue
            v = iou(b, gb)
            if v > best:
                best, bi = v, j
        if best >= 0.5:
            used.add(bi)
            outcomes.append(("TP", c, cf, b, best))
            continue
        other = max([(iou(b, gb), gc) for j, (gc, gb) in enumerate(gt) if gc != c], default=(0, None))
        if other[0] >= 0.5:
            outcomes.append(("wrong_class", c, cf, b, other[0]))
        elif best >= 0.1:
            outcomes.append(("poor_box", c, cf, b, best))
        else:
            outcomes.append(("false_positive", c, cf, b, best))
    missed = [j for j in range(len(gt)) if j not in used]
    return outcomes, missed


def draw(ax, im, boxes, names, title, gt=False):
    ax.imshow(im)
    for item in boxes:
        if gt:
            c, b = item
            label = names[c]
        else:
            kind, c, cf, b, _ = item
            label = f"{names[c]} {cf:.2f}"
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2.5,
                                       ec=COLORS[c % 4], ls="-" if gt or item[0] == "TP" else "--"))
        ax.text(b[0], b[1], label, color="white", fontsize=10, bbox=dict(fc=COLORS[c % 4], alpha=0.85, pad=1, lw=0))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "datasets" / "african-wildlife-clean.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for shown predictions")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    names, ds = cfg["names"], Path(cfg["path"])
    out = ROOT / "results" / f"{args.split}_predictions"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    imgs = sorted((ds / "images" / args.split).iterdir())
    per_image, records = [], {}
    totals = {"TP": 0, "wrong_class": 0, "poor_box": 0, "false_positive": 0, "missed": 0, "gt": 0}
    for i in range(0, len(imgs), 16):
        for res in model.predict([str(p) for p in imgs[i:i + 16]], imgsz=args.imgsz, conf=args.conf,
                                 iou=args.iou, verbose=False):
            p = Path(res.path)
            h, w = res.orig_shape
            gt = load_gt(ds / "labels" / args.split / (p.stem + ".txt"), w, h)
            preds = [(int(c), float(cf), b) for b, c, cf in
                     zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
            outcomes, missed = match(preds, gt)
            cnt = {k: sum(o[0] == k for o in outcomes) for k in ("TP", "wrong_class", "poor_box", "false_positive")}
            cnt["missed"] = len(missed)
            cnt["gt"] = len(gt)
            for k in totals:
                totals[k] += cnt[k]
            max_fp_conf = max([o[2] for o in outcomes if o[0] != "TP"], default=0)
            per_image.append({"image": p.name, **cnt, "max_conf_of_wrong_pred": round(max_fp_conf, 3)})
            records[p.name] = (gt, outcomes, missed)

    with open(out / "per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_image[0]))
        wr.writeheader()
        wr.writerows(per_image)

    n_pred = totals["TP"] + totals["wrong_class"] + totals["poor_box"] + totals["false_positive"]
    summary = {
        "weights": args.weights, "split": args.split, "imgsz": args.imgsz,
        "conf_threshold": args.conf, "nms_iou": args.iou, "match_iou": 0.5, **totals,
        "precision_at_conf": round(totals["TP"] / n_pred, 4) if n_pred else None,
        "recall_at_conf": round(totals["TP"] / totals["gt"], 4) if totals["gt"] else None,
        "images_with_any_error": sum(1 for r in per_image if r["missed"] or r["wrong_class"]
                                     or r["poor_box"] or r["false_positive"]),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # --- prediction grid: 6 random test images ---
    rnd = random.Random(args.seed)
    sample = rnd.sample(list(records), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, name in zip(axes.flat, sample):
        gt, outcomes, _ = records[name]
        draw(ax, Image.open(ds / "images" / args.split / name).convert("RGB"), outcomes, names, name)
    plt.suptitle(f"Test predictions (conf >= {args.conf})", fontsize=15)
    plt.tight_layout()
    plt.savefig(out / "predictions_grid.jpg", dpi=75)
    plt.close()

    # --- failure galleries ---
    keys = {"missed": lambda r: r["missed"], "false_positive": lambda r: r["false_positive"],
            "wrong_class": lambda r: r["wrong_class"], "poor_box": lambda r: r["poor_box"]}
    failure_list = {}
    for kind, f in keys.items():
        cands = sorted([r for r in per_image if f(r) > 0], key=lambda r: (-f(r), -r["max_conf_of_wrong_pred"]))[:4]
        failure_list[kind] = [c["image"] for c in cands]
        if not cands:
            continue
        fig, axes = plt.subplots(len(cands), 2, figsize=(12, 4.2 * len(cands)), squeeze=False)
        for row, c in zip(axes, cands):
            gt, outcomes, missed = records[c["image"]]
            im = Image.open(ds / "images" / args.split / c["image"]).convert("RGB")
            draw(row[0], im, gt, names, f"GROUND TRUTH  {c['image']}", gt=True)
            draw(row[1], im, outcomes, names,
                 f"PREDICTION  missed={c['missed']} FP={c['false_positive']} "
                 f"wrong_cls={c['wrong_class']} poor_box={c['poor_box']}")
        plt.tight_layout()
        plt.savefig(out / f"failures_{kind}.jpg", dpi=65)
        plt.close()
    json.dump(failure_list, open(out / "failure_candidates.json", "w"), indent=2)
    print("Failure candidates:", json.dumps(failure_list, indent=2))


if __name__ == "__main__":
    main()
