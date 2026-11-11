import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

LENGTHS = [0, 4, 8, 16, 32]
DIRECTIONS = ["horizontal", "vertical"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def line_kernel(length: int, direction: str) -> np.ndarray:
    """Averaging kernel along a line: the image a camera takes while moving `length` px."""
    kernel = np.zeros((length, length), dtype=np.float32)
    if direction == "horizontal":
        kernel[length // 2, :] = 1
    else:
        kernel[:, length // 2] = 1
    return kernel / kernel.sum()


def load(image: Path, imgsz: int) -> tuple[np.ndarray, float]:
    """The image resized so that its long side is imgsz, and the scale factor used."""
    bgr = cv2.imread(str(image))
    if bgr is None:
        raise FileNotFoundError(image)
    scale = imgsz / max(bgr.shape[:2])
    size = (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("runs/yolo26s-1280/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("data/fsoco5/test/images"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", type=Path, default=Path("runs/phase1/blur"))
    args = parser.parse_args()

    images = sorted(args.images.iterdir())
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    settings = [("none", 0)] + [(d, n) for d in DIRECTIONS for n in LENGTHS if n > 0]

    for direction, length in settings:
        path = args.out / f"predictions_{direction}_{length}.csv"
        if path.exists():
            print(f"{path} exists, skipping")
            continue
        partial = path.with_suffix(".partial")
        with open(partial, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "cls", "conf", "x0", "y0", "x1", "y1"])
            for start in range(0, len(images), args.batch):
                chunk = images[start : start + args.batch]
                loaded = [load(i, args.imgsz) for i in chunk]
                arrays = [
                    cv2.filter2D(a, -1, line_kernel(length, direction)) if length else a
                    for a, _ in loaded
                ]
                results = model.predict(
                    source=arrays,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )
                for image, (_, scale), result in zip(chunk, loaded, results, strict=True):
                    assert isinstance(result, Results)
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    xyxy = (np.asarray(boxes.xyxy.tolist()) / scale).tolist()
                    for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), xyxy):
                        writer.writerow(
                            [image.stem, int(cls), f"{conf:.5f}", *(f"{v:.1f}" for v in box)]
                        )
        partial.rename(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
