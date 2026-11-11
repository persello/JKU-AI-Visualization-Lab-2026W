import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

RATE = 5  # frames per second of video


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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", type=Path, default=Path("runs/phase1/videos"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        step = max(1, round(capture.get(cv2.CAP_PROP_FPS) / RATE))
        path = args.out / f"{video.stem}_detections.csv"
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "cls", "conf", "x0", "y0", "x1", "y1"])
            frame_number, batch, numbers = 0, [], []
            while True:
                ok = capture.grab()
                if not ok:
                    break
                if frame_number % step == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        batch.append(frame)
                        numbers.append(frame_number)
                frame_number += 1
                if len(batch) == args.batch or (batch and not ok):
                    write(model, batch, numbers, writer, args)
                    batch, numbers = [], []
            if batch:
                write(model, batch, numbers, writer, args)
        capture.release()
        print(f"wrote {path} ({frame_number} frames, every {step}th)")


def write(model: YOLO, batch: list, numbers: list[int], writer, args: argparse.Namespace) -> None:
    results = model.predict(
        source=batch, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
    )
    for number, result in zip(numbers, results, strict=True):
        assert isinstance(result, Results)
        boxes = result.boxes
        if boxes is None:
            continue
        for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            writer.writerow([number, int(cls), f"{conf:.4f}", *(f"{v:.1f}" for v in box)])


if __name__ == "__main__":
    main()
