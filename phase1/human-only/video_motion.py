import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import size_analysis as sa
import video_detect as vd
import video_stability as vs

FLOW_WIDTH = 320
SKIP = {"aPPk06QK_ME": 8.0}  # seconds of title animation at the start
STILL = 50  # px/s at the model input: below this, the car is taken to stand
GRID_STILL = "#ecebe6"
# Hand-picked example moments (seconds): standing, then driving, with near cones in view.
FRAMES = {"aPPk06QK_ME": (12.0, 20.0), "o5vES5QaeiQ": (0.5, 100.0), "CrISno91wPg": (3.0, 15.0)}


def plot_frames(
    video: Path,
    detections: pd.DataFrame,
    tracked: pd.DataFrame,
    fps: float,
    label: str,
    figures: Path,
) -> None:
    """Two frames side by side, standing and driving, with the model's detections in white."""
    fig, axes = plt.subplots(1, 2, figsize=(sa.FIGSIZE[0], sa.FIGSIZE[1] * 0.62))
    capture = cv2.VideoCapture(str(video))
    sampled = np.sort(detections["frame"].unique())
    for ax, seconds, state in zip(axes, FRAMES[video.stem], ["standing", "driving"]):
        # The nearest frame that video_detect.py ran on.
        frame_number = int(sampled[np.argmin(np.abs(sampled - seconds * fps))])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        ax.imshow(frame[..., ::-1])
        shown = detections[detections["frame"] == frame_number]
        kept = int((tracked["frame"] == frame_number).sum())
        for row in shown.itertuples():
            ax.add_patch(
                plt.Rectangle(
                    (row.x0, row.y0),
                    row.x1 - row.x0,
                    row.y1 - row.y0,
                    fill=False,
                    edgecolor="#ffffff",
                    linewidth=1.2,
                )
            )
        ax.set_axis_off()
        ax.set_title(
            f"{state}, {frame_number / fps:.0f} s: {len(shown)} detected, tracker keeps {kept}",
            loc="left",
            color=sa.INK_2,
            fontsize=11,
        )
    capture.release()
    fig.suptitle(label, x=0.01, ha="left", fontsize=12, color=sa.INK)
    fig.tight_layout()
    fig.savefig(figures / f"video_frames_{video.stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def image_motion(video: Path) -> tuple[np.ndarray, float]:
    """Per frame: the 90th percentile of optical-flow length, in model-input px per second."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    to_input = sa.IMGSZ / max(width, height) * width / FLOW_WIDTH
    motion, previous = [], None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (FLOW_WIDTH, round(height * FLOW_WIDTH / width)))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if previous is None:
            motion.append(np.nan)
        else:
            start = np.zeros((*grey.shape, 2), dtype=np.float32)  # ignored without the flag
            flow = cv2.calcOpticalFlowFarneback(previous, grey, start, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion.append(float(np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 90)))
        previous = grey
    capture.release()
    return np.asarray(motion) * to_input * fps, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--tracks", type=Path, default=Path("runs/phase1/videos"))
    parser.add_argument("--out", type=Path, default=Path("phase1/human-only"))
    args = parser.parse_args()

    per_second = []
    for video in vs.VIDEOS:
        cache = args.tracks / f"{video}.motion.npy"  # the flow takes a minute; keep it
        fps = float((args.tracks / f"{video}.info").read_text().split()[0])
        if cache.exists():
            motion = np.load(cache)
        else:
            motion, fps = image_motion(args.videos / f"{video}.mp4")
            np.save(cache, motion)
        tracked = pd.read_csv(args.tracks / f"{video}.csv")
        detections = pd.read_csv(args.tracks / f"{video}_detections.csv")
        frames = pd.DataFrame(
            {
                "motion": motion,
                "tracked": tracked.groupby("frame")
                .size()
                .reindex(range(len(motion)), fill_value=0),
            }
        )
        # video_detect.py ran on every step-th frame; the others have no detection count.
        step = max(1, round(fps / vd.RATE))
        counts = detections.groupby("frame").size()
        sampled = np.arange(0, len(frames), step)
        frames["detected"] = counts.reindex(sampled, fill_value=0).reindex(frames.index)
        frames["second"] = (np.arange(len(frames)) / fps).astype(int)
        frames = frames[frames["second"] >= SKIP.get(video, 0)]
        s = frames.groupby("second").agg(
            detected=("detected", "mean"), tracked=("tracked", "mean"), motion=("motion", "median")
        )
        per_second.append(s.reset_index().assign(video=video))
        rho = s["detected"].rank().corr(s["motion"].rank())  # Spearman, without scipy
        print(f"{vs.VIDEOS[video]}: {len(s)} s, Spearman rho (detected, motion) = {rho:.2f}")
    table = pd.concat(per_second, ignore_index=True)
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "video_motion.csv", index=False, float_format="%.3f")

    sa.style()
    # Each video keeps its own axes: the comparison is within a video, standing against driving.
    fig, axes = plt.subplots(2, len(vs.VIDEOS), figsize=sa.FIGSIZE, sharex="col")
    for column, (video, label) in enumerate(vs.VIDEOS.items()):
        t = table[table["video"] == video]
        top, bottom = axes[0, column], axes[1, column]
        # Shade the seconds in which the picture is almost still: the car stands.
        still = (t["motion"] < STILL).to_numpy()
        seconds = t["second"].to_numpy()
        for ax in (top, bottom):
            for s in seconds[still]:
                ax.axvspan(s - 0.5, s + 0.5, color=GRID_STILL, linewidth=0, zorder=0)
        top.plot(seconds, t["detected"], color=sa.INK, linewidth=1.5, label="detected")
        top.plot(seconds, t["tracked"], color=sa.MUTED, linewidth=1.5, label="kept by the tracker")
        bottom.plot(seconds, t["motion"], color="#6250d6", linewidth=1.5)
        bottom.fill_between(seconds, 0, t["motion"], color="#6250d6", alpha=0.1, linewidth=0)
        top.set_title(label.replace(", ", "\n"), loc="left", color=sa.INK, fontsize=12)
        bottom.set_xlabel("Time (s)")
        top.set_ylim(bottom=0)
        bottom.set_ylim(bottom=0)
        moving, standing = t[~still], t[still]
        print(
            f"  {label}: per frame standing ({len(standing)} s) {standing['detected'].mean():.1f}"
            f" detected, {standing['tracked'].mean():.1f} tracked; driving ({len(moving)} s)"
            f" {moving['detected'].mean():.1f} detected, {moving['tracked'].mean():.1f} tracked"
        )
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[0, 0].set_ylabel("Cones per frame")
    axes[1, 0].set_ylabel("Image motion\n(px/s at input)")
    fig.tight_layout()
    fig.text(
        0.01,
        -0.02,
        f"Shaded: image motion under {STILL} px/s, the car stands.",
        ha="left",
        va="top",
        color=sa.INK_2,
        fontsize=11,
    )
    figures = args.out / "figures"
    sa.save(fig, figures, "video_motion")
    for video, label in vs.VIDEOS.items():
        fps = float((args.tracks / f"{video}.info").read_text().split()[0])
        tracked = pd.read_csv(args.tracks / f"{video}.csv")
        detections = pd.read_csv(args.tracks / f"{video}_detections.csv")
        plot_frames(args.videos / f"{video}.mp4", detections, tracked, fps, label, figures)
    print(f"wrote {figures / 'video_motion'} and {results / 'video_motion.csv'}")


if __name__ == "__main__":
    main()
