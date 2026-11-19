"""Evaluate a checkpoint on one split and save the metrics as JSON.

Usage:
    python scripts/evaluate.py --weights runs/baseline/weights/best.pt --split val --imgsz 640 --name baseline
    python scripts/evaluate.py --weights weights/best.pt --split test --imgsz 640 --name final

Settings (recorded in the JSON):
  conf = 0.001  -> keep almost all predictions so the full precision-recall curve can be built (standard for mAP)
  iou  = 0.7    -> IoU threshold used by NMS to remove duplicate boxes
  A prediction is a true positive at mAP@0.50 if IoU with a same-class GT box >= 0.50.
  Ultralytics reports P and R at the confidence threshold that maximises mean F1 over classes.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "datasets" / "african-wildlife-clean.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  batch=args.batch, plots=True, project=str(ROOT / "runs" / "eval"),
                  name=f"{args.name}_{args.split}", exist_ok=True, verbose=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    out = {
        "run": args.name,
        "split": args.split,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "eval_conf_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "tp_iou_threshold_for_mAP50": 0.5,
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
        "plots_dir": str(Path(m.save_dir).relative_to(ROOT)),
    }
    dest = ROOT / "results" / "metrics"
    dest.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest / f"{args.name}_{args.split}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
