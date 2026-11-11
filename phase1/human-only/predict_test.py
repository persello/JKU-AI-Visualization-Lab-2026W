import argparse
import csv
from collections.abc import Iterator
from pathlib import Path

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
    parser.add_argument("--weights", type=Path, default=Path("runs/yolo26s-1280/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("data/fsoco5/test/images"))
    parser.add_argument("--imgsz", type=int, default=1280, help="the training image size")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", type=Path, default=Path("runs/phase1/test"))
    args = parser.parse_args()

    images = sorted(args.images.iterdir())
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    with (
        open(args.out / "predictions.csv", "w", newline="") as pred_file,
        open(args.out / "images.csv", "w", newline="") as image_file,
    ):
        preds = csv.writer(pred_file)
        preds.writerow(["image", "cls", "conf", "x0", "y0", "x1", "y1"])
        sizes = csv.writer(image_file)
        sizes.writerow(["image", "width", "height"])

        # A list source is run as one batch, so pass the images a few at a time.
        def results() -> Iterator[Results]:
            for start in range(0, len(images), args.batch):
                chunk = [str(i) for i in images[start : start + args.batch]]
                for result in model.predict(
                    source=chunk,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                ):
                    assert isinstance(result, Results)
                    yield result

        for n, (image, result) in enumerate(zip(images, results(), strict=True), start=1):
            height, width = result.orig_shape
            sizes.writerow([image.stem, width, height])
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.tolist()
                for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), xyxy):
                    preds.writerow(
                        [image.stem, int(cls), f"{conf:.5f}", *(f"{v:.1f}" for v in box)]
                    )
            if n % 100 == 0:
                print(f"{n}/{len(images)} images")
    print(f"wrote {args.out / 'predictions.csv'} and {args.out / 'images.csv'}")


if __name__ == "__main__":
    main()
