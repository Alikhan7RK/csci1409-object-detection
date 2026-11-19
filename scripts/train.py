"""Train one run from a YAML config and record everything needed for the report.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/exp_imgsz320.yaml
    python scripts/train.py --config configs/baseline.yaml --epochs 1 --fraction 0.05   # smoke test

Outputs: runs/<name>/ (Ultralytics logs, results.csv, plots, weights/best.pt)
         results/<name>/run_info.json  (versions, hardware, settings, training time)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override (smoke tests only)")
    ap.add_argument("--fraction", type=float, help="override: use a fraction of train set (smoke tests only)")
    ap.add_argument("--imgsz", type=int, help="override (smoke tests only)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg.pop("model")
    cfg["data"] = str((ROOT / cfg["data"]).resolve())
    cfg["project"] = str((ROOT / cfg["project"]).resolve())
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.fraction:
        cfg["fraction"] = args.fraction
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.device is not None:
        cfg["device"] = args.device

    model = YOLO(model_name)
    t0 = time.time()
    model.train(**cfg)
    train_seconds = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    info = {
        "run": cfg["name"],
        "config_file": args.config,
        "framework": f"ultralytics {ultralytics.__version__}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
        "python": platform.python_version(),
        "initial_weights": model_name,
        "train_settings": {k: cfg[k] for k in ("imgsz", "batch", "epochs", "optimizer", "lr0", "lrf",
                                               "momentum", "weight_decay", "seed", "deterministic")},
        "fraction": cfg.get("fraction", 1.0),
        "training_time_seconds": round(train_seconds, 1),
        "training_time_minutes": round(train_seconds / 60, 2),
        "save_dir": str(save_dir.relative_to(ROOT)),
        "best_checkpoint": str((save_dir / "weights" / "best.pt").relative_to(ROOT)),
        "checkpoint_selection": "Ultralytics best.pt = epoch with the highest validation fitness "
                                "(in ultralytics 8.4 fitness = validation mAP@0.5:0.95)",
    }
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / "run_info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
