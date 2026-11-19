"""Generates notebooks/object_detection_colab.ipynb (kept as a script so the notebook is easy to review)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# CSCI1409 Bonus Project — African Wildlife Object Detection (YOLO11n)

Fine-tunes a COCO-pretrained **YOLO11n** on the **African Wildlife** dataset (buffalo, elephant, rhino, zebra),
runs one controlled experiment (**image size 640 → 320**), evaluates the selected model on the held-out test set
and analyses failure cases.

**Before running:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
Total time on a free T4 is roughly 30–40 minutes. Everything starts from a fresh runtime.
""")

md("## 0. Environment")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU - training will be very slow"
""")
code("""
# Change this URL only if you forked the repository
REPO_URL = "https://github.com/Alikhan7RK/csci1409-object-detection.git"

import os
if not os.path.exists("/content/csci1409-object-detection"):
    !git clone -q {REPO_URL} /content/csci1409-object-detection
%cd /content/csci1409-object-detection
""")
code("""
!pip install -q -r requirements.txt
import ultralytics, torch, platform
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| python", platform.python_version())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

md("""
## 1. Dataset
Download the official split, then inspect it: images/objects per split, label sanity checks and a
near-duplicate search (perceptual hashes).
""")
code("""
!python scripts/prepare_data.py
!python scripts/data_report.py --data datasets/african-wildlife.yaml
""")
md("""
The official split contains **the same photographs in train and val/test** (cross-split duplicates → data
leakage). `make_clean_split.py` groups duplicates and moves each cross-split group to *train*
(deterministic rule, no randomness). We then re-run the report to confirm **0 cross-split duplicates**.
""")
code("""
!python scripts/make_clean_split.py
!python scripts/data_report.py --data datasets/african-wildlife-clean.yaml
""")
code("""
from IPython.display import Image, Markdown, display
import json, pandas as pd
display(Markdown(open("results/data/african-wildlife-clean/stats.md").read()))
display(pd.read_csv("results/data/african-wildlife/duplicates.csv").head(10))
display(Image("results/data/african-wildlife-clean/gt_samples.jpg", width=1100))
""")
md("""
### Missing / wrong label check
A COCO-pretrained YOLO11m already knows *elephant* and *zebra*. Confident COCO detections with no matching
ground-truth box are **candidates** for missing labels (they must be verified by eye).
""")
code("""
!python scripts/check_missing_labels.py --data datasets/african-wildlife-clean.yaml
display(pd.read_csv("results/data/label_check_candidates.csv").head(15))
display(Image("results/data/label_check_candidates.jpg", width=1100))
""")

md("""
## 2. Run A — baseline (imgsz 640)
Settings are in `configs/baseline.yaml` (YOLO11n COCO weights, 40 epochs, batch 16, SGD lr0=0.01, seed 42).
""")
code("""
!python scripts/train.py --config configs/baseline.yaml
display(Image("runs/baseline/results.png", width=1100))
""")

md("""
## 3. Run B — controlled experiment (imgsz 320)
**Hypothesis (written before training):** the animals are large in most images (median box covers 15–32 % of the
image), so halving the input resolution to 320 should lower validation mAP@0.50 only slightly (≤ 3 points),
while making training noticeably faster. Any loss should come mostly from small / distant animals and herds.

Only `imgsz` changes; data split, initial checkpoint, seed, epochs, batch and learning rate are identical.
""")
code("""
!python scripts/train.py --config configs/exp_imgsz320.yaml
display(Image("runs/exp_imgsz320/results.png", width=1100))
""")

md("## 4. Compare both runs on the validation set and choose the final run")
code("""
!python scripts/evaluate.py --weights runs/baseline/weights/best.pt --split val --imgsz 640 --name baseline
!python scripts/evaluate.py --weights runs/exp_imgsz320/weights/best.pt --split val --imgsz 320 --name exp_imgsz320
!python scripts/compare_runs.py
display(Markdown(open("results/comparison.md").read()))
display(Image("results/learning_curves_comparison.png", width=1100))
""")

md("""
## 5. Final evaluation on the held-out TEST set
The final run is chosen **only** from validation results (`results/final_run.json`). The test set is used once.
""")
code("""
final = json.load(open("results/final_run.json")); print(final)
!python scripts/evaluate.py --weights {final["weights"]} --split test --imgsz {final["imgsz"]} --name final
""")
code("""
!python scripts/predict_and_failures.py --weights {final["weights"]} --split test --imgsz {final["imgsz"]} --conf 0.25
display(Image("results/test_predictions/predictions_grid.jpg", width=1100))
""")
code("""
import os
for kind in ["missed", "false_positive", "wrong_class", "poor_box"]:
    p = f"results/test_predictions/failures_{kind}.jpg"
    if os.path.exists(p):
        display(Markdown(f"### Failure type: {kind}")); display(Image(p, width=900))
    else:
        print("No", kind, "failures at conf 0.25")
""")

md("## 6. Save and download all outputs")
code("""
!python scripts/collect_outputs.py
from google.colab import files
files.download("outputs.zip")
""")

md("""
## 7. (Optional) Use the trained checkpoint without training
Downloads `weights/best.pt` from the repository and predicts on a few test images.
""")
code("""
WEIGHTS_URL = REPO_URL.replace(".git", "").replace("github.com", "raw.githubusercontent.com") + "/main/weights/best.pt"
!wget -q -O /content/best_downloaded.pt {WEIGHTS_URL} && echo downloaded
from ultralytics import YOLO
import glob
m = YOLO("/content/best_downloaded.pt")
for r in m.predict(sorted(glob.glob("datasets/african-wildlife-clean/images/test/*"))[:3], conf=0.25):
    display(Image(data=__import__("cv2").imencode(".jpg", r.plot())[1].tobytes(), width=500))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:  # nbformat wants a list of lines
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = ROOT / "notebooks" / "object_detection_colab.ipynb"
out.parent.mkdir(exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("Wrote", out)
