import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import size_analysis as sa
from matplotlib.colors import LinearSegmentedColormap

VIDEOS = {
    "o5vES5QaeiQ": "Chalmers, trackdrive",
    "CrISno91wPg": "KA-RaceIng, autocross",
    "aPPk06QK_ME": "BCN eMotorsport, skidpad",
}
VIDEO_COLOURS = ["#6250d6", "#1baf7a", "#e34948"]  # validated set, as for the outcome chart
VIDEO_MARKERS = ["o", "^", "D"]
MIN_DETECTIONS = 10
SPEED_EDGES = np.array([0, 50, 100, 200, 400, np.inf])  # model-input px per second
MIN_TRACKS = 10  # cells and bins with fewer tracks are not drawn


def load_tracks(folder: Path, video: str) -> tuple[pd.DataFrame, float]:
    boxes = pd.read_csv(folder / f"{video}.csv")
    fps, _, width, height = (float(v) for v in (folder / f"{video}.info").read_text().split())
    scale = sa.IMGSZ / max(width, height)
    boxes["height_in"] = (boxes["y1"] - boxes["y0"]) * scale
    boxes["cx"] = (boxes["x0"] + boxes["x1"]) / 2 * scale
    boxes["cy"] = (boxes["y0"] + boxes["y1"]) / 2 * scale
    return boxes, fps


def summarise(boxes: pd.DataFrame, fps: float) -> pd.DataFrame:
    rows = []
    for track, t in boxes.groupby("track"):
        t = t.sort_values("frame").drop_duplicates("frame")
        if len(t) < MIN_DETECTIONS:
            continue
        frames = t["frame"].to_numpy()
        gaps = np.diff(frames) - 1
        step = np.hypot(np.diff(t["cx"]), np.diff(t["cy"])) / np.maximum(np.diff(frames), 1)
        rows.append(
            {
                "track": track,
                "cls": sa.CLASSES[int(t["cls"].mode().iloc[0])],
                "first": frames[0] / fps,
                "seconds": (frames[-1] - frames[0] + 1) / fps,
                "detections": len(t),
                "coverage": len(t) / (frames[-1] - frames[0] + 1),
                "dropouts": int((gaps > 0).sum()),
                "height": t["height_in"].median(),
                "speed": float(np.median(step)) * fps,
                "conf": t["conf"].median(),
            }
        )
    tracks = pd.DataFrame(rows)
    tracks["bin"] = np.digitize(tracks["height"], sa.EDGES[1:-1])
    tracks["speed_bin"] = np.digitize(tracks["speed"], SPEED_EDGES[1:-1])
    return tracks


def plot_stability(tracks: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=sa.FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for (video, label), colour, marker in zip(VIDEOS.items(), VIDEO_COLOURS, VIDEO_MARKERS):
        t = tracks[tracks["video"] == video]
        g = t.groupby("bin").agg(
            n=("coverage", "size"), coverage=("coverage", "mean"), height=("height", "median")
        )
        g = g[g["n"] >= MIN_TRACKS]
        ax.plot(
            g["height"],
            g["coverage"],
            color=colour,
            marker=marker,
            markersize=7,
            markeredgecolor=sa.SURFACE,
            markeredgewidth=1,
            label=f"{label} ({len(t)} tracks)",
        )
    sa.log_height_axis(ax, "Cone height at the model input (px)", ticks=[4, 8, 16, 32, 64, 128])
    ax.set_xlim(3.5, 200)
    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("Frames detected while tracked")
    ax.legend(loc="lower right", fontsize=11)

    ax = axes[1]
    grid = tracks.pivot_table(
        index="speed_bin", columns="bin", values="coverage", aggfunc=["mean", "size"]
    )
    mean, size = grid["mean"], grid["size"]
    mean = mean.where(size >= MIN_TRACKS)
    bins = [b for b in range(len(sa.EDGES) - 1) if b in mean.columns and mean[b].notna().any()]
    speeds = list(range(len(SPEED_EDGES) - 1))
    values = mean.reindex(index=speeds, columns=bins).to_numpy(dtype=float)
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
    ramp.set_bad(sa.SURFACE)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=ramp, vmin=0.6, vmax=1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                light = values[i, j] < 0.8
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=sa.INK if light else "#ffffff",
                )
    ax.set_xticks(range(len(bins)), [sa.bin_label(b) for b in bins], rotation=45, ha="right")
    speed_labels = [f"{SPEED_EDGES[i]:.0f}–{SPEED_EDGES[i + 1]:.0f}" for i in speeds[:-1]]
    ax.set_yticks(speeds, [*speed_labels, f"≥{SPEED_EDGES[-2]:.0f}"])
    ax.set_xlabel("Cone height at the model input (px)")
    ax.set_ylabel("Image speed (px/s)")
    ax.grid(False)
    ax.set_title("All videos: share of frames detected", loc="left", color=sa.INK, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    sa.save(fig, figures, "video_stability")


def runs(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each run of consecutive frames."""
    breaks = np.flatnonzero(np.diff(frames) > 1)
    return np.r_[frames[0], frames[breaks + 1]], np.r_[frames[breaks], frames[-1]] + 1


def plot_timeline(
    boxes: pd.DataFrame, fps: float, label: str, figures: Path, seconds: float = 5.0
) -> None:
    """Every track in the window of `seconds` with the most dropouts, one row per track."""
    per_track = {track: np.sort(t["frame"].unique()) for track, t in boxes.groupby("track")}
    # Frames at which a dropout starts, over all tracks.
    gap_frames = np.concatenate(
        [ends[:-1] for _, ends in (runs(f) for f in per_track.values()) if len(ends) > 1]
        or [np.array([], dtype=int)]
    )
    width = int(seconds * fps)
    last = int(boxes["frame"].max())
    counts = [
        ((gap_frames >= s) & (gap_frames < s + width)).sum() for s in range(0, last, width // 5)
    ]
    start = int(np.argmax(counts)) * (width // 5)
    window = (start, start + width)

    shown = {track: f for track, f in per_track.items() if f[0] < window[1] and f[-1] >= window[0]}
    order = {track: i for i, track in enumerate(sorted(shown, key=lambda k: shown[k][0]))}
    cls_of = boxes.groupby("track")["cls"].agg(lambda s: int(s.mode().iloc[0]))
    fig, ax = plt.subplots(figsize=sa.FIGSIZE)
    for track, frames in shown.items():
        starts, ends = runs(frames)
        ax.broken_barh(
            [(s / fps, (e - s) / fps) for s, e in zip(starts, ends)],
            (order[track] - 0.35, 0.7),
            color=sa.CLASS_COLOURS[cls_of[track]],
            linewidth=0,
        )
    ax.set_xlim(window[0] / fps, window[1] / fps)
    ax.set_xlabel("Time in the video (s)")
    ax.set_ylabel(f"{len(order)} tracks, by first appearance")
    ax.set_ylim(-1, len(order))
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in sa.CLASS_COLOURS]
    ax.legend(handles, sa.CLASSES, loc="upper left", ncol=5, fontsize=11, bbox_to_anchor=(0, 1.12))
    ax.set_title(
        f"{label}, the {seconds:.0f} s with the most dropouts: a bar is a cone while detected,"
        " a gap in a row is a dropout",
        loc="left",
        color=sa.INK_2,
        fontsize=11,
        pad=30,
    )
    fig.tight_layout()
    sa.save(fig, figures, "video_timeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, default=Path("runs/phase1/videos"))
    parser.add_argument("--out", type=Path, default=Path("phase1/human-only"))
    parser.add_argument("--timeline", default="CrISno91wPg", help="video for the timeline")
    args = parser.parse_args()

    all_tracks, loaded = [], {}
    for video, label in VIDEOS.items():
        if not (args.tracks / f"{video}.info").exists():
            print(f"no tracks for {video}, skipping")
            continue
        boxes, fps = load_tracks(args.tracks, video)
        tracks = summarise(boxes, fps).assign(video=video)
        loaded[video] = (boxes, tracks, fps)
        all_tracks.append(tracks)
        print(
            f"{label}: {len(tracks)} tracks, median {tracks['seconds'].median():.1f} s, "
            f"coverage {tracks['coverage'].mean():.3f}, "
            f"dropouts per track-second {tracks['dropouts'].sum() / tracks['seconds'].sum():.2f}"
        )
    tracks = pd.concat(all_tracks, ignore_index=True)
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(results / "video_tracks.csv", index=False, float_format="%.4f")
    by_size = tracks.groupby("bin").agg(
        tracks=("coverage", "size"),
        coverage=("coverage", "mean"),
        dropouts_per_s=("dropouts", "sum"),
        seconds=("seconds", "sum"),
    )
    by_size["dropouts_per_s"] /= by_size.pop("seconds")
    by_size.index = [sa.bin_label(b) for b in by_size.index]
    print(by_size.round(3).to_string())

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sa.style()
    plot_stability(tracks, figures)
    if args.timeline in loaded:
        boxes, _, fps = loaded[args.timeline]
        plot_timeline(boxes, fps, VIDEOS[args.timeline], figures)
    print(f"wrote figures to {figures} and {results / 'video_tracks.csv'}")


if __name__ == "__main__":
    main()
