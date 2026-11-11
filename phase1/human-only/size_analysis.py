import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, NullLocator

CLASSES = ["blue", "yellow", "orange", "large orange", "unknown"]
IMGSZ = 1280
THRESHOLD = 0.25
IOU = 0.5
NEAR_IOU = 0.1  # a prediction this close to a cone counts as aimed at it
MIN_CONES = 20  # bins with fewer labelled cones are not drawn
EDGES = np.array([0, *2 ** np.arange(2.5, 7.01, 0.5), np.inf])  # half-octave bins from 5.7 px
OUTCOMES = ["detected", "box off", "wrong class", "missed"]

# Colours: the cone's own colour for its class; a validated set for the other categories.
CLASS_COLOURS = ["#2a78d6", "#c98500", "#eb6834", "#b4441b", "#6f6e69"]
OUTCOME_COLOURS = ["#c3c2b7", "#1baf7a", "#6250d6", "#e34948"]
TEAM_COLOURS = ["#6250d6", "#e87ba4", "#1baf7a", "#e34948"]
TEAM_MARKERS = ["o", "s", "^", "D"]
# The content area of a slide in slides/template (24.8 x 11.5 cm), so the text keeps its size.
FIGSIZE = (9.8, 4.5)
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#ffffff",
)


def load_images(path: Path) -> pd.DataFrame:
    images = pd.read_csv(path)
    images["team"] = images["image"].str.split("_").str[0]
    images["scale"] = IMGSZ / images[["width", "height"]].max(axis=1)
    return images


def load_ground_truth(labels: Path, images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image, width, height in images[["image", "width", "height"]].itertuples(index=False):
        label = labels / f"{image}.txt"
        if not label.exists():
            continue
        for line in label.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:])
            rows.append(
                (
                    image,
                    int(parts[0]),
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                )
            )
    return pd.DataFrame(rows, columns=["image", "cls", "x0", "y0", "x1", "y1"])


def add_size(boxes: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    boxes = boxes.merge(images[["image", "team", "scale"]], on="image", how="left")
    boxes["height_px"] = boxes["y1"] - boxes["y0"]
    boxes["height_in"] = boxes["height_px"] * boxes["scale"]
    boxes["bin"] = np.digitize(boxes["height_in"], EDGES[1:-1])
    return boxes


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N x 4, xyxy) with every box in b (M x 4)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Add the matched prediction and the outcome to gt, and the matched label and kind to pred."""
    gt_boxes = gt[["x0", "y0", "x1", "y1"]].to_numpy()
    pred_boxes = pred[["x0", "y0", "x1", "y1"]].to_numpy()
    gt_cls, pred_cls = gt["cls"].to_numpy(), pred["cls"].to_numpy()
    conf = pred["conf"].to_numpy()

    gt_pred = np.full(len(gt), -1)
    pred_gt = np.full(len(pred), -1)
    outcome = np.full(len(gt), "missed", dtype=object)
    confused = np.full(len(gt), -1)
    kind = np.full(len(pred), "below threshold", dtype=object)
    gt_by_image = gt.groupby("image").indices
    no_rows = np.array([], dtype=int)

    for image, pi in pred.groupby("image").indices.items():
        gi = gt_by_image.get(image, no_rows)
        pi = pi[np.argsort(-conf[pi], kind="stable")]
        ious = iou_matrix(pred_boxes[pi], gt_boxes[gi])
        same = pred_cls[pi][:, None] == gt_cls[gi][None, :]
        free = np.ones(len(gi), dtype=bool)
        for k in range(len(pi)):
            if not free.any():
                break
            candidates = np.where(same[k] & free, ious[k], -1.0)
            j = int(candidates.argmax())
            if candidates[j] >= IOU:
                free[j] = False
                gt_pred[gi[j]] = pi[k]
                pred_gt[pi[k]] = gi[j]

        # Outcomes at the operating threshold.
        kept = conf[pi] >= THRESHOLD
        ious_kept, same_kept = ious[kept], same[kept]
        detected = (gt_pred[gi] >= 0) & (conf[np.maximum(gt_pred[gi], 0)] >= THRESHOLD)
        wrong_class = ((ious_kept >= IOU) & ~same_kept).any(axis=0)
        box_off = ((ious_kept >= NEAR_IOU) & same_kept).any(axis=0)
        outcome[gi] = np.select(
            [detected, wrong_class, box_off], ["detected", "wrong class", "box off"], "missed"
        )
        pk = pi[kept]
        if len(pk) and len(gi):
            # The class given instead: that of the most overlapping box of another class.
            best = np.where(~same_kept, ious_kept, -1.0).argmax(axis=0)
            confused[gi] = np.where(wrong_class & ~detected, pred_cls[pk][best], -1)
        kind[pk] = np.select(
            [
                pred_gt[pk] >= 0,
                ((ious_kept >= IOU) & same_kept).any(axis=1),
                ((ious_kept >= IOU) & ~same_kept).any(axis=1),
                (ious_kept >= NEAR_IOU).any(axis=1),
            ],
            ["correct", "duplicate", "wrong class", "box off"],
            "background",
        )

    gt["pred"] = gt_pred
    gt["match_conf"] = np.where(gt_pred >= 0, conf[np.maximum(gt_pred, 0)], np.nan)
    gt["outcome"] = outcome
    gt["given_as"] = confused
    pred["gt"] = pred_gt
    pred["kind"] = kind


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: 101-point interpolated area under the precision-recall curve."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    hits = tp[order]
    ctp, cfp = np.cumsum(hits), np.cumsum(~hits)
    recall = ctp / n_gt
    precision = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1))[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(
        np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0).mean()
    )


def ap50_by_bin(gt: pd.DataFrame, pred: pd.DataFrame, cls: int) -> list[float]:
    """AP50 per height bin; a correct prediction takes its label's bin, a wrong one its own."""
    p = pred[pred["cls"] == cls]
    tp = p["gt"].to_numpy() >= 0
    p_bin = np.where(tp, gt["bin"].to_numpy()[np.maximum(p["gt"].to_numpy(), 0)], p["bin"])
    out = []
    for b in range(len(EDGES) - 1):
        n_gt = int(((gt["cls"] == cls) & (gt["bin"] == b)).sum())
        keep = p_bin == b
        out.append(average_precision(p["conf"].to_numpy()[keep], tp[keep], n_gt))
    return out


def binned_rate(
    rows: pd.DataFrame, hit: np.ndarray, images: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Rate of hit per height bin, with a 95 % interval from resampling whole images."""
    n_bins = len(EDGES) - 1
    code = pd.Categorical(rows["image"], categories=images["image"]).codes
    n = np.zeros((len(images), n_bins))
    k = np.zeros((len(images), n_bins))
    np.add.at(n, (code, rows["bin"].to_numpy()), 1)
    np.add.at(k, (code, rows["bin"].to_numpy()), hit.astype(float))
    draws = rng.integers(0, len(images), (1000, len(images)))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty bins give NaN
        boot = k[draws].sum(axis=1) / n[draws].sum(axis=1)
        rate = k.sum(axis=0) / n.sum(axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    median = rows.groupby("bin")["height_in"].median().reindex(range(n_bins))
    return pd.DataFrame(
        {"n": n.sum(axis=0).astype(int), "rate": rate, "lo": lo, "hi": hi, "height": median}
    )


# Figures ------------------------------------------------------------------------------------


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
        }
    )


def log_height_axis(
    ax: Axes,
    label: str = "Cone height at the model input (px)",
    ticks: tuple[int, ...] | list[int] = (4, 8, 16, 32, 64, 128, 256),
) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(3.5, 300)
    ax.set_xlabel(label)


def rate_line(
    ax: Axes, rate: pd.DataFrame, colour: str, label: str | None = None, marker: str = "o"
) -> None:
    r = rate[rate["n"] >= MIN_CONES]
    ax.fill_between(r["height"], r["lo"], r["hi"], color=colour, alpha=0.12, linewidth=0)
    ax.plot(
        r["height"],
        r["rate"],
        color=colour,
        marker=marker,
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label=label,
    )


def save(fig: plt.Figure, figures: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(figures / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recall_by_size(
    overall: pd.DataFrame, per_class: list[pd.DataFrame], figures: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    panels = [("All cones", INK, overall), *zip(CLASSES, CLASS_COLOURS, per_class)]
    for ax, (name, colour, rate) in zip(axes.flat, panels):
        if rate is not overall:
            r = overall[overall["n"] >= MIN_CONES]
            ax.plot(r["height"], r["rate"], color=AXIS, linewidth=1.5, zorder=1)
        rate_line(ax, rate, colour)
        ax.set_title(f"{name.capitalize()}  ({rate['n'].sum():,})", loc="left", color=INK)
        log_height_axis(ax, "", ticks=[4, 8, 16, 32, 64, 128])
        ax.set_xlim(3.5, 200)
        ax.set_ylim(0, 1.02)
    fig.supxlabel("Cone height at the model input (px); number of labelled cones in brackets")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall")
    axes[0, 1].text(
        0.98, 0.06, "grey: all cones", transform=axes[0, 1].transAxes, ha="right", color=MUTED
    )
    fig.tight_layout()
    save(fig, figures, "recall_by_size")


def plot_outcome_by_size(gt: pd.DataFrame, figures: Path) -> None:
    counts = pd.crosstab(gt["bin"], gt["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
    counts = counts[counts.sum(axis=1) >= MIN_CONES]
    shares = counts.div(counts.sum(axis=1), axis=0)
    labels = [bin_label(b) for b in shares.index]
    x = np.arange(len(shares))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom = np.zeros(len(shares))
    gap = 0.006  # thin surface gap between stacked segments
    for outcome, colour in zip(OUTCOMES, OUTCOME_COLOURS):
        values = shares[outcome].to_numpy()
        ax.bar(
            x,
            np.clip(values - gap, 0, None),
            bottom=bottom,
            width=0.6,
            color=colour,
            label=outcome,
            linewidth=0,
        )
        bottom += values
    for xi, n in zip(x, counts.sum(axis=1)):
        ax.text(float(xi), 1.02, f"{n:,}", ha="center", va="bottom", color=MUTED, fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Cone height at the model input (px); number of labelled cones above each bar")
    ax.set_ylabel("Share of labelled cones")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", visible=False)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, figures, "outcome_by_size")


def plot_confidence_by_size(gt: pd.DataFrame, figures: Path) -> None:
    found = gt.dropna(subset=["match_conf"])
    q = found.groupby("bin").agg(
        n=("match_conf", "size"),
        height=("height_in", "median"),
        median=("match_conf", "median"),
        lo=("match_conf", lambda s: s.quantile(0.25)),
        hi=("match_conf", lambda s: s.quantile(0.75)),
    )
    q = q[q["n"] >= MIN_CONES]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(q["height"], q["lo"], q["hi"], color=INK, alpha=0.08, linewidth=0)
    ax.plot(
        q["height"],
        q["median"],
        color=INK,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
    )
    ax.axhline(THRESHOLD, color=OUTCOME_COLOURS[3], linewidth=1.5)
    ax.text(3.7, THRESHOLD + 0.015, f"threshold {THRESHOLD}", color=INK_2, va="bottom")
    ax.text(
        q["height"].iloc[-1],
        q["hi"].iloc[-1] + 0.02,
        "median, band: middle 50 %",
        ha="right",
        va="bottom",
        color=INK_2,
    )
    log_height_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Confidence of correct boxes")
    fig.tight_layout()
    save(fig, figures, "confidence_by_size")


def plot_recall_by_team(
    gt: pd.DataFrame, images: pd.DataFrame, rng: np.random.Generator, figures: Path
) -> None:
    teams = sorted(gt["team"].unique())
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, column, label in [
        (axes[0], "height_in", "Cone height at the model input (px)"),
        (axes[1], "height_px", "Cone height in the original image (px)"),
    ]:
        # The lines meet near 1, so the teams are told apart by marker shape and the legend.
        for team, colour, marker in zip(teams, TEAM_COLOURS, TEAM_MARKERS):
            rows = gt[gt["team"] == team].copy()
            rows["height_in"] = rows[column]
            rows["bin"] = np.digitize(rows[column], EDGES[1:-1])
            rate = binned_rate(rows, rows["outcome"].to_numpy() == "detected", images, rng)
            n_images = int((images["team"] == team).sum())
            rate_line(ax, rate, colour, label=f"{team} ({n_images} images)", marker=marker)
        log_height_axis(ax, label)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Recall")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, figures, "recall_by_team")


def bin_label(b: int) -> str:
    lo, hi = EDGES[b], EDGES[b + 1]
    if lo == 0:
        return f"<{hi:.0f}"
    if np.isinf(hi):
        return f"≥{lo:.0f}"
    return f"{lo:.0f}–{hi:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=Path("runs/phase1/test"))
    parser.add_argument("--labels", type=Path, default=Path("data/fsoco5/test/labels"))
    parser.add_argument("--out", type=Path, default=Path("phase1/human-only"))
    args = parser.parse_args()

    images = load_images(args.predictions / "images.csv")
    gt = add_size(load_ground_truth(args.labels, images), images)
    pred = add_size(pd.read_csv(args.predictions / "predictions.csv"), images)
    match(gt, pred)
    rng = np.random.default_rng(0)

    detected = gt["outcome"].to_numpy() == "detected"
    overall = binned_rate(gt, detected, images, rng)
    per_class = [
        binned_rate(gt[gt["cls"] == c], detected[gt["cls"] == c], images, rng)
        for c in range(len(CLASSES))
    ]
    kept = pred[pred["conf"] >= THRESHOLD]
    precision = kept.groupby("bin")["kind"].apply(lambda s: (s == "correct").mean())

    # Tables.
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    ap = np.array([ap50_by_bin(gt, pred, c) for c in range(len(CLASSES))])
    # The mean AP of a bin uses only the classes with enough labelled cones in it.
    enough = np.array([r["n"].to_numpy() >= MIN_CONES for r in per_class])
    by_size = overall.assign(
        bin=[bin_label(b) for b in range(len(EDGES) - 1)],
        precision=precision.reindex(range(len(EDGES) - 1)).to_numpy(),
        map50=np.nanmean(np.where(enough, ap, np.nan), axis=0),
    )
    outcomes = pd.crosstab(gt["bin"], gt["outcome"]).reindex(
        index=range(len(EDGES) - 1), columns=OUTCOMES, fill_value=0
    )
    by_size = pd.concat([by_size, outcomes.reset_index(drop=True)], axis=1)
    by_size.to_csv(results / "by_size.csv", index=False, float_format="%.4f")
    by_class = pd.concat(
        [
            r.assign(cls=CLASSES[c], bin=[bin_label(b) for b in range(len(EDGES) - 1)], ap50=ap[c])
            for c, r in enumerate(per_class)
        ]
    )
    by_class.to_csv(results / "by_size_class.csv", index=False, float_format="%.4f")
    wrong = gt[gt["outcome"] == "wrong class"]
    confusion = pd.crosstab(
        wrong["cls"].map(dict(enumerate(CLASSES))).rename("labelled"),
        wrong["given_as"].map(dict(enumerate(CLASSES))).rename("given as"),
    )
    confusion.to_csv(results / "wrong_class.csv")

    # Summary on the console.
    all_ap = [
        average_precision(
            pred.loc[pred["cls"] == c, "conf"].to_numpy(),
            pred.loc[pred["cls"] == c, "gt"].to_numpy() >= 0,
            int((gt["cls"] == c).sum()),
        )
        for c in range(len(CLASSES))
    ]
    print(f"{len(images)} images, {len(gt)} labelled cones, {len(kept)} boxes >= {THRESHOLD}")
    print(
        "AP50 per class:", dict(zip(CLASSES, np.round(all_ap, 3))), f"mAP50 {np.mean(all_ap):.3f}"
    )
    print(f"recall {detected.mean():.3f}, precision {(kept['kind'] == 'correct').mean():.3f}")
    print(by_size.round(3).to_string(index=False))
    print(kept["kind"].value_counts().to_string())
    print("labelled cones given the wrong class:\n" + confusion.to_string())
    for c in range(len(CLASSES)):
        rows = gt[gt["cls"] == c]
        print(
            f"{CLASSES[c]}: median height {rows['height_in'].median():.1f} px, "
            f"wrong class median {rows.loc[rows['outcome'] == 'wrong class', 'height_in'].median():.1f} px"
        )

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    plot_recall_by_size(overall, per_class, figures)
    plot_outcome_by_size(gt, figures)
    plot_confidence_by_size(gt, figures)
    plot_recall_by_team(gt, images, rng, figures)
    print(f"wrote figures to {figures} and tables to {results}")


if __name__ == "__main__":
    main()
