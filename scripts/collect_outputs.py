"""Copy the important training artefacts from runs/ (git-ignored) into results/ (committed),
copy the final checkpoint to weights/best.pt and zip everything into outputs.zip for download.
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["results.csv", "results.png", "args.yaml", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "labels.jpg",
        "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg", "val_batch1_pred.jpg"]


def main():
    for run_dir in sorted((ROOT / "runs").glob("*")):
        if run_dir.name == "eval" or not run_dir.is_dir():
            continue
        dest = ROOT / "results" / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in KEEP:
            if (run_dir / f).exists():
                shutil.copy2(run_dir / f, dest / f)
    for ev in sorted((ROOT / "runs" / "eval").glob("*")):
        dest = ROOT / "results" / "eval_plots" / ev.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ev.glob("*.png"):
            shutil.copy2(f, dest / f.name)

    final = json.load(open(ROOT / "results" / "final_run.json"))
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy2(ROOT / final["weights"], ROOT / "weights" / "best.pt")
    print("Final checkpoint ->", ROOT / "weights" / "best.pt")

    zpath = ROOT / "outputs.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("results", "weights"):
            for f in (ROOT / folder).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT))
        for run_dir in (ROOT / "runs").glob("*/weights/best.pt"):  # both runs' best checkpoints
            z.write(run_dir, run_dir.relative_to(ROOT))
    print("Wrote", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
