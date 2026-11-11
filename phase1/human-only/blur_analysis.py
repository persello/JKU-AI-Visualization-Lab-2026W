import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import blur_test as bt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import size_analysis as sa

GROUPS = [("small, < 11 px", 0, 11), ("medium, 11–32 px", 11, 32), ("large, ≥ 32 px", 32, np.inf)]
GROUP_COLOURS = ["#b4b3ad", "#6f6e69", "#0b0b0b"]  # ordinal: lighter is smaller
GROUP_MARKERS = ["o", "s", "D"]
LENGTHS = [0, 4, 8, 16, 32]
DIRECTIONS = ["horizontal", "vertical"]


def plot_examples(image: Path, centre: tuple[float, float], figures: Path) -> None:
    """A 2 x 5 grid: the same crop (model-input pixels) under every blur setting."""
    picture, scale = bt.load(image, sa.IMGSZ)
    cx, cy = int(centre[0] * scale), int(centre[1] * scale)
    w, h = 200, 92
    x0 = int(np.clip(cx - w // 2, 0, picture.shape[1] - w))
    y0 = int(np.clip(cy - h // 2, 0, picture.shape[0] - h))
    fig, axes = plt.subplots(2, len(LENGTHS), figsize=(sa.FIGSIZE[0], 2.9))
    for row, direction in zip(axes, DIRECTIONS):
        for ax, length in zip(row, LENGTHS):
            blurred = (
                cv2.filter2D(picture, -1, bt.line_kernel(length, direction)) if length else picture
            )
            crop = blurred[y0 : y0 + h, x0 : x0 + w, ::-1]  # BGR to RGB
            ax.imshow(crop, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{length} px" if length else "no blur", fontsize=12, color=sa.INK_2)
        row[0].set_ylabel(direction, color=sa.INK)
    fig.suptitle(
        f"{image.stem}: a {w} × {h} px crop at the model input",
        x=0.01,
        ha="left",
        fontsize=11,
        color=sa.INK_2,
    )
    fig.tight_layout()
    fig.savefig(figures / "blur_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame) -> list[dict]:
    gt, pred = gt.copy(), pred.copy()
    sa.match(gt, pred)
    kept = pred[pred["conf"] >= sa.THRESHOLD]
    rows = []
    for name, lo, hi in [("all", 0, np.inf), *GROUPS]:
        g = gt[(gt["height_in"] >= lo) & (gt["height_in"] < hi)]
        p = kept[(kept["height_in"] >= lo) & (kept["height_in"] < hi)]
        rows.append(
            {
                "group": name,
                "cones": len(g),
                "recall": (g["outcome"] == "detected").mean(),
                "missed": (g["outcome"] == "missed").mean(),
                "precision": (p["kind"] == "correct").mean(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=Path("runs/phase1/blur"))
    parser.add_argument("--images", type=Path, default=Path("runs/phase1/test/images.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/fsoco5/test/labels"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/fsoco5/test/images"))
    parser.add_argument("--example", default="amz_amz_00723", help="test image for the examples")
    parser.add_argument("--out", type=Path, default=Path("phase1/human-only"))
    args = parser.parse_args()

    images = sa.load_images(args.images)
    gt = sa.add_size(sa.load_ground_truth(args.labels, images), images)

    rows = []
    for direction in DIRECTIONS:
        for length in LENGTHS:
            name = "none_0" if length == 0 else f"{direction}_{length}"
            path = args.predictions / f"predictions_{name}.csv"
            if not path.exists():
                print(f"{path} missing, skipping")
                continue
            pred = sa.add_size(pd.read_csv(path), images)
            for row in evaluate(gt, pred):
                rows.append({"direction": direction, "length": length, **row})
    table = pd.DataFrame(rows)
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "blur.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))

    sa.style()
    fig, axes = plt.subplots(1, 2, figsize=sa.FIGSIZE, sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        for (group, _, _), colour, marker in zip(GROUPS, GROUP_COLOURS, GROUP_MARKERS):
            t = table[(table["direction"] == direction) & (table["group"] == group)]
            x = [LENGTHS.index(n) for n in t["length"]]
            ax.plot(
                x,
                t["recall"],
                color=colour,
                marker=marker,
                markersize=7,
                markeredgecolor=sa.SURFACE,
                markeredgewidth=1,
                label=group,
            )
        ax.set_xticks(range(len(LENGTHS)), [str(n) for n in LENGTHS])
        ax.set_xlabel(f"{direction.capitalize()} blur length (px at the model input)")
        ax.set_title(f"{direction.capitalize()} blur", loc="left", color=sa.INK)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall")
    axes[1].legend(loc="lower left", title="Cone height", title_fontsize=12, alignment="left")
    fig.tight_layout()
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sa.save(fig, figures, "blur_effect")

    # Centre the example crop on the median position of that image's mid-sized cones.
    cones = gt[(gt["image"] == args.example) & gt["height_in"].between(8, 30)]
    centre = ((cones["x0"] + cones["x1"]).median() / 2, (cones["y0"] + cones["y1"]).median() / 2)
    plot_examples(next(args.images_dir.glob(f"{args.example}.*")), centre, figures)
    print(f"wrote {figures / 'blur_effect'} and {results / 'blur.csv'}")


if __name__ == "__main__":
    main()
