import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import size_analysis as sa
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

SMALL = 11  # px at the model input
FALSE_COLOUR = "#e87ba4"
DETECTED_COLOUR = "#ffffff"  # the chart's grey is hard to see on asphalt


def crop_box(cones: pd.DataFrame, width: int, height: int, aspect: float) -> tuple[int, ...]:
    """A crop with the given aspect ratio around the cones, inside the image."""
    x0, x1 = cones["x0"].min(), cones["x1"].max()
    y0, y1 = cones["y0"].min(), cones["y1"].max()
    w = max(x1 - x0, (y1 - y0) * aspect) * 1.3
    w = min(max(w, width * 0.45), width, height * aspect)
    h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - w / 2, 0, width - w))
    top = int(np.clip(cy - h / 2, 0, height - h))
    return left, top, int(left + w), int(top + h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=Path("runs/phase1/test"))
    parser.add_argument("--data", type=Path, default=Path("data/fsoco5/test"))
    parser.add_argument("--out", type=Path, default=Path("phase1/human-only/figures"))
    args = parser.parse_args()

    images = sa.load_images(args.predictions / "images.csv")
    gt = sa.add_size(sa.load_ground_truth(args.data / "labels", images), images)
    pred = sa.add_size(pd.read_csv(args.predictions / "predictions.csv"), images)
    sa.match(gt, pred)
    false = pred[(pred["conf"] >= sa.THRESHOLD) & (pred["kind"] != "correct")]
    sa.style()

    small_misses = gt[(gt["height_in"] < SMALL) & (gt["outcome"] != "detected")]
    colours = dict(zip(sa.OUTCOMES, sa.OUTCOME_COLOURS)) | {"detected": DETECTED_COLOUR}
    figsize = (sa.FIGSIZE[0], sa.FIGSIZE[1] - 0.5)  # room for the legend below
    for team, misses in small_misses.groupby("team"):
        image = misses["image"].value_counts().index[0]
        cones = gt[gt["image"] == image]
        info = images.set_index("image").loc[image]
        left, top, right, bottom = crop_box(
            misses[misses["image"] == image],
            int(info["width"]),
            int(info["height"]),
            figsize[0] / figsize[1],
        )
        photo = Image.open(next((args.data / "images").glob(f"{image}.*"))).convert("RGB")

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(photo.crop((left, top, right, bottom)), extent=(left, right, bottom, top))
        line = 1.2  # points; thin, so that boxes of a few pixels stay outlines
        for row in cones.itertuples():
            ax.add_patch(
                Rectangle(
                    (row.x0, row.y0),
                    row.x1 - row.x0,
                    row.y1 - row.y0,
                    fill=False,
                    edgecolor=colours[str(row.outcome)],
                    linewidth=line if row.outcome == "detected" else line * 1.5,
                )
            )
        for row in false[false["image"] == image].itertuples():
            ax.add_patch(
                Rectangle(
                    (row.x0, row.y0),
                    row.x1 - row.x0,
                    row.y1 - row.y0,
                    fill=False,
                    edgecolor=FALSE_COLOUR,
                    linewidth=line,
                    linestyle=(0, (2, 1)),
                )
            )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_axis_off()
        counts = cones["outcome"].value_counts()
        handles = [Line2D([], [], color=c, linewidth=3) for c in colours.values()] + [
            Line2D([], [], color=FALSE_COLOUR, linewidth=3, linestyle=(0, (2, 1)))
        ]
        labels = [f"{o} ({counts.get(o, 0)})" for o in colours] + [
            f"false detection ({int((false['image'] == image).sum())})"
        ]
        # A dark legend, so that the white "detected" key shows as it does on the photo.
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.07),
            frameon=True,
            facecolor="#383835",
            edgecolor="none",
            labelcolor="#f0efec",
            fontsize=11,
        )
        legend.get_frame().set_alpha(1)
        ax.set_title(
            f"{team}: {image}, cropped; counts are for the whole image",
            loc="left",
            color=sa.INK_2,
            fontsize=11,
        )
        fig.tight_layout()
        path = args.out / f"example_{team}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
