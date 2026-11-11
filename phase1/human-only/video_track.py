import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--weights", type=Path, default=Path("runs/yolo26s-1280/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", type=Path, default=Path("runs/phase1/videos"))
    args = parser.parse_args()

    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        print(f"{video.name}: {frames} frames at {fps:.0f} fps, {width}x{height}")

        model = YOLO(args.weights)  # a fresh model, so that no track carries over between videos
        path = args.out / f"{video.stem}.csv"
        args.out.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "time", "track", "cls", "conf", "x0", "y0", "x1", "y1"])
            results = model.track(
                source=str(video),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                stream=True,
                save=True,
                project=str(args.out.resolve()),
                name=video.stem,
                exist_ok=True,
                verbose=False,
            )
            for frame, result in enumerate(results):
                assert isinstance(result, Results)
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue
                rows = zip(
                    boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                )
                for track, cls, conf, box in rows:
                    writer.writerow(
                        [
                            frame,
                            f"{frame / fps:.3f}",
                            int(track),
                            int(cls),
                            f"{conf:.4f}",
                            *(f"{v:.1f}" for v in box),
                        ]
                    )
                if frame % 500 == 0:
                    print(f"  frame {frame}/{frames}")
        with open(args.out / f"{video.stem}.info", "w") as file:
            file.write(f"{fps} {frames} {width} {height}\n")
        print(f"wrote {path} and the annotated video in {args.out / video.stem}")


if __name__ == "__main__":
    main()
