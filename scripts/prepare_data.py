"""Download FSOCO in YOLO format from Kaggle and build a 5-class copy for training.

The Kaggle dataset (persello/fsoco-yolo, about 20 GB) has 10 classes: each cone colour appears
twice, once from the bounding-box part of FSOCO and once (prefixed `seg_`) from the segmentation
part. This script merges them into 5 classes and turns polygon labels into boxes. Images are
symlinked, not copied, so the 5-class copy needs only a few MB.
"""

import argparse
import os
from pathlib import Path

import kagglehub
import yaml

DATASET = "persello/fsoco-yolo"
CLASSES = [
    "blue_cone",
    "yellow_cone",
    "orange_cone",
    "large_orange_cone",
    "unknown_cone",
]
SPLITS = {"train": "train", "val": "validate", "test": "test"}


def to_box(values: list[float]) -> list[float]:
    """Normalised (x, y, w, h) of a box row, or the bounding box of a polygon row."""
    if len(values) == 4:
        return values
    xs, ys = values[0::2], values[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def convert_split(src: Path, dst: Path, old_names: list[str]) -> int:
    """Symlink the images of one split and write remapped labels; returns the number of images."""
    remap = {i: CLASSES.index(name.removeprefix("seg_")) for i, name in enumerate(old_names)}
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted((src / "images").iterdir()):
        link = dst / "images" / image.name
        if not link.exists():
            os.symlink(image.resolve(), link)
        rows = []
        label = src / "labels" / f"{image.stem}.txt"
        if label.exists():
            for line in label.read_text().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                box = to_box([float(v) for v in parts[1:]])
                rows.append(f"{remap[int(parts[0])]} " + " ".join(f"{v:.6f}" for v in box))
        (dst / "labels" / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/fsoco5"), help="output folder")
    args = parser.parse_args()

    root = Path(kagglehub.dataset_download(DATASET)) / "fsoco-yolo"
    old = yaml.safe_load((root / "data.yaml").read_text())
    out = args.out.resolve()
    for split, folder in SPLITS.items():
        n = convert_split(root / folder, out / folder, old["names"])
        print(f"{split}: {n} images")

    config = {"path": str(out), **SPLITS, "names": dict(enumerate(CLASSES))}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
