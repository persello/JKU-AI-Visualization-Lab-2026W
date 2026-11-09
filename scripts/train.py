"""Fine-tune a YOLO26 detector on the 5-class FSOCO copy made by prepare_data.py."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="pretrained YOLO26 weights (n, s, m, l, x)",
    )
    parser.add_argument("--data", type=Path, default=Path("data/fsoco5/data.yaml"))
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="training image size; small far cones need a large one",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="stop after this many hours (overrides --epochs)",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--name",
        default=None,
        help="run name under runs/ (default: model and image size)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        time=args.hours,
        batch=args.batch,
        device=args.device,
        project=str(Path("runs").resolve()),
        name=args.name or f"{Path(args.model).stem}-{args.imgsz}",
        patience=30,
        seed=0,
    )


if __name__ == "__main__":
    main()
