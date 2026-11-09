"""Download FSOCO in YOLO format from Kaggle and build a 5-class copy for training.

The Kaggle dataset (persello/fsoco-yolo, about 20 GB) is our 2024 conversion of the FSOCO
bounding-box set. It has 10 classes because it took the class list from FSOCO's meta.json, which
also lists the `seg_` classes of the segmentation set. This script merges them into 5 classes and
turns any polygon labels into boxes.

The 2024 split put each image into train, val or test at random (6534 / 1174 / 1215 images), so
all 41 teams appear in every split. Consecutive frames of one team's recording are nearly
identical, so test images have near-copies in train. By
default this script splits again by team (the image name starts with the team folder): about 80 %
of the images for training, 10 % for validation and 10 % for testing, with every team in only one
split. Use `--split original` to keep the 2024 split.

Images are symlinked, not copied, so the 5-class copy takes only a few MB.
"""

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import kagglehub
import yaml

DATASET = "persello/fsoco-yolo"
CLASSES = ["blue_cone", "yellow_cone", "orange_cone", "large_orange_cone", "unknown_cone"]
SPLITS = {"train": "train", "val": "validate", "test": "test"}
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def to_box(values: list[float]) -> list[float]:
    """Normalised (x, y, w, h) of a box row, or the bounding box of a polygon row."""
    if len(values) == 4:
        return values
    xs, ys = values[0::2], values[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def team_of(image: Path) -> str:
    return image.stem.split("_")[0]


def split_by_team(images: list[Path], seed: int) -> dict[str, list[Path]]:
    """Assign whole teams to train, val and test so that each split gets about its fraction."""
    teams = sorted({team_of(i) for i in images})
    random.Random(seed).shuffle(teams)
    sizes = Counter(team_of(i) for i in images)
    targets = {s: f * len(images) for s, f in FRACTIONS.items()}
    assigned: dict[str, str] = {}
    filled = Counter()
    for team in teams:
        # Give the team to the split that is furthest below its target.
        split = max(FRACTIONS, key=lambda s: (targets[s] - filled[s]) / targets[s])
        assigned[team] = split
        filled[split] += sizes[team]
    return {s: [i for i in images if assigned[team_of(i)] == s] for s in FRACTIONS}


def write_split(images: list[Path], dst: Path, remap: dict[int, int], counts: Counter) -> None:
    """Symlink the images of one split and write their labels with the 5 merged classes."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for image in images:
        link = dst / "images" / image.name
        if not link.exists():
            os.symlink(image.resolve(), link)
        rows = []
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if label.exists():
            for line in label.read_text().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = remap[int(parts[0])]
                counts[CLASSES[cls]] += 1
                box = to_box([float(v) for v in parts[1:]])
                rows.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
        (dst / "labels" / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/fsoco5"), help="output folder")
    parser.add_argument("--split", choices=["team", "original"], default="team")
    parser.add_argument("--seed", type=int, default=0, help="seed for the team split")
    args = parser.parse_args()

    root = Path(kagglehub.dataset_download(DATASET)) / "fsoco-yolo"
    old_names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    remap = {i: CLASSES.index(name.removeprefix("seg_")) for i, name in enumerate(old_names)}

    original = {s: sorted((root / f / "images").iterdir()) for s, f in SPLITS.items()}
    if args.split == "original":
        splits = original
    else:
        splits = split_by_team([i for images in original.values() for i in images], args.seed)

    out = args.out.resolve()
    for split, images in splits.items():
        counts: Counter = Counter()
        write_split(images, out / SPLITS[split], remap, counts)
        teams = len({team_of(i) for i in images})
        print(f"{split}: {len(images)} images, {teams} teams, boxes {dict(counts)}")

    config = {"path": str(out), **SPLITS, "names": dict(enumerate(CLASSES))}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
