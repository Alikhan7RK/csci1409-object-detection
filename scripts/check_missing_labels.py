"""Find candidate MISSING or WRONG labels using a COCO-pretrained detector as a second opinion.

COCO already contains 'elephant' and 'zebra' (not buffalo or rhino). For every image we run the
COCO model and flag:
  * possible_missing_label : COCO model sees an elephant/zebra (conf >= --conf) that overlaps
                             no ground-truth box of ANY class (IoU < --iou)
  * possible_wrong_class   : that detection overlaps (IoU >= 0.5) a GT box of the other COCO-known class
                             (elephant <-> zebra). Overlaps with rhino/buffalo boxes are ignored because
                             COCO has no such classes and labels rhinos as "elephant".
Flags are candidates only; they must be checked by eye (see results/data/label_check_candidates.jpg).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_TO_OURS = {20: "elephant", 22: "zebra"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_gt(lbl, w, h, names):
    out = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
                out.append((names[c], [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "african-wildlife-clean.yaml"))
    ap.add_argument("--model", default="yolo11m.pt", help="COCO-pretrained model used as reviewer")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "results" / "data"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    names, ds = cfg["names"], Path(cfg["path"])
    model = YOLO(args.model)
    flags = []
    for split in args.splits:
        imgs = sorted((ds / "images" / split).iterdir())
        # predict in small chunks: passing the whole list at once loads every image into RAM
        results = (r for i in range(0, len(imgs), 16)
                   for r in model.predict([str(p) for p in imgs[i:i + 16]], conf=args.conf,
                                          classes=list(COCO_TO_OURS), verbose=False))
        for res in results:
            p = Path(res.path)
            h, w = res.orig_shape
            gt = load_gt(ds / "labels" / split / (p.stem + ".txt"), w, h, names)
            for box, c, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                cname = COCO_TO_OURS[int(c)]
                best_same = max([iou(box, g) for n, g in gt if n == cname], default=0)
                if best_same >= args.iou:
                    continue
                # COCO has no rhino/buffalo class, so it often calls a rhino "elephant" or a buffalo
                # nothing. Only an elephant<->zebra disagreement can be a real wrong-class label.
                known = [(iou(box, g), n) for n, g in gt if n != cname and n in COCO_TO_OURS.values()]
                best_known = max(known, default=(0, None))
                best_any = max([iou(box, g) for _, g in gt], default=0)
                if best_known[0] >= 0.5:
                    kind, detail = "possible_wrong_class", f"GT says {best_known[1]}"
                elif best_any >= args.iou:
                    continue  # overlaps a rhino/buffalo box: reviewer limitation, not a label error
                else:
                    kind, detail = "possible_missing_label", f"max IoU with any GT box {best_any:.2f}"
                flags.append([split, p.name, kind, cname, round(cf, 3), round(best_same, 3), detail,
                              *[round(v) for v in box]])
    flags.sort(key=lambda r: -r[4])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "label_check_candidates.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "flag", "coco_class", "coco_conf", "best_iou_with_gt_same_class",
                     "detail", "x1", "y1", "x2", "y2"])
        wr.writerows(flags)

    # visualise top candidates: green = ground truth, red dashed = COCO detection that has no GT
    top = flags[:12]
    if top:
        cols = 4
        rows = int(np.ceil(len(top) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, fl in zip(axes.flat, top):
            split, name = fl[0], fl[1]
            im = Image.open(ds / "images" / split / name).convert("RGB")
            w, h = im.size
            ax.imshow(im)
            for n, g in load_gt(ds / "labels" / split / (Path(name).stem + ".txt"), w, h, names):
                ax.add_patch(patches.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="lime", lw=2))
                ax.text(g[0], g[1], n, color="black", fontsize=8, bbox=dict(fc="lime", pad=1, lw=0))
            x1, y1, x2, y2 = fl[7:]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=2, ls="--"))
            ax.set_title(f"{split}/{name}\n{fl[2]}: COCO {fl[3]} {fl[4]:.2f}", fontsize=9)
        plt.tight_layout()
        plt.savefig(out / "label_check_candidates.jpg", dpi=70)
    from collections import Counter
    print(f"{len(flags)} candidate boxes:", dict(Counter(f[2] for f in flags)))
    print("Saved", out / "label_check_candidates.csv")


if __name__ == "__main__":
    main()
