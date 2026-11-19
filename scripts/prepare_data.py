"""Download the African Wildlife dataset and write a dataset YAML with an absolute path.

Usage:
    python scripts/prepare_data.py            # downloads into ./datasets/african-wildlife
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

import yaml

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "datasets"), help="where to put the dataset")
    args = ap.parse_args()

    root = Path(args.root)
    ds = root / "african-wildlife"
    if not (ds / "images" / "train").exists():
        root.mkdir(parents=True, exist_ok=True)
        zpath = root / "african-wildlife.zip"
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ds)
        zpath.unlink()
    else:
        print("Dataset already present:", ds)

    # Template in configs/ has a relative path; write a resolved copy for Ultralytics.
    cfg = yaml.safe_load(open(ROOT / "configs" / "african-wildlife.yaml"))
    cfg["path"] = str(ds.resolve())
    out = ROOT / "datasets" / "african-wildlife.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    for split in ("train", "val", "test"):
        n = len(list((ds / "images" / split).glob("*")))
        print(f"{split:5s}: {n} images")
    print("Wrote", out)


if __name__ == "__main__":
    main()
