"""Compare the two runs on the VALIDATION set and choose the final run.

Reads results/<run>/run_info.json and results/metrics/<run>_val.json for each run,
writes results/comparison.md, results/learning_curves_comparison.png and results/final_run.json.
Selection rule: highest validation mAP@0.50; tie (< 0.005) -> higher mAP@0.50:0.95. Test data is NOT used.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline", "exp_imgsz320"])
    args = ap.parse_args()

    rows, infos = [], {}
    for r in args.runs:
        info = json.load(open(ROOT / "results" / r / "run_info.json"))
        met = json.load(open(ROOT / "results" / "metrics" / f"{r}_val.json"))
        infos[r] = info
        rows.append({
            "Run": r,
            "Changed setting": f"imgsz = {info['train_settings']['imgsz']}",
            "Val precision": met["precision"],
            "Val recall": met["recall"],
            "Val mAP@0.50": met["mAP50"],
            "Val mAP@0.50:0.95": met["mAP50_95"],
            "Training time (min)": info["training_time_minutes"],
            "Inference (ms/img)": met["speed_ms_per_image"].get("inference"),
        })
    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False)
    (ROOT / "results" / "comparison.md").write_text(md + "\n")
    print(md)

    a, b = rows[0], rows[1]
    if abs(a["Val mAP@0.50"] - b["Val mAP@0.50"]) < 0.005:
        best = max(rows, key=lambda x: x["Val mAP@0.50:0.95"])
    else:
        best = max(rows, key=lambda x: x["Val mAP@0.50"])
    final = {"final_run": best["Run"], "imgsz": infos[best["Run"]]["train_settings"]["imgsz"],
             "weights": infos[best["Run"]]["best_checkpoint"],
             "rule": "highest validation mAP@0.50 (ties within 0.005 broken by mAP@0.50:0.95)"}
    json.dump(final, open(ROOT / "results" / "final_run.json", "w"), indent=2)
    print("Final run:", final)

    # learning curves of both runs on one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for r in args.runs:
        csv = ROOT / infos[r]["save_dir"] / "results.csv"
        d = pd.read_csv(csv)
        d.columns = [c.strip() for c in d.columns]
        axes[0].plot(d["epoch"], d["train/box_loss"], label=f"{r} train")
        axes[0].plot(d["epoch"], d["val/box_loss"], "--", label=f"{r} val")
        axes[1].plot(d["epoch"], d["metrics/mAP50(B)"], label=r)
        axes[2].plot(d["epoch"], d["metrics/precision(B)"], label=f"{r} P")
        axes[2].plot(d["epoch"], d["metrics/recall(B)"], "--", label=f"{r} R")
    for ax, t in zip(axes, ["Box loss", "Validation mAP@0.50", "Validation precision / recall"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "learning_curves_comparison.png", dpi=90)


if __name__ == "__main__":
    main()
