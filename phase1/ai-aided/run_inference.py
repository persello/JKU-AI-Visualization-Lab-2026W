from pathlib import Path

import pandas as pd
from ultralytics import YOLO

CLASS_NAMES = {
    0: "blue",
    1: "yellow",
    2: "orange",
    3: "large orange",
    4: "unknown",
}

PREDICTION_COLUMNS = [
    "image_name",
    "class_id",
    "class_name",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_height_px",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "yolo26s-fsoco5.pt"
    requested_test_dir = project_root / "data" / "fsoco5" / "images" / "test"
    existing_test_dir = project_root / "data" / "fsoco5" / "test" / "images"
    output_path = Path(__file__).resolve().parent / "predictions.csv"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if requested_test_dir.is_dir():
        test_images_dir = requested_test_dir
    elif existing_test_dir.is_dir():
        test_images_dir = existing_test_dir
        print(
            "Requested test image directory was not found; "
            f"using the repository dataset directory: {test_images_dir}"
        )
    else:
        raise FileNotFoundError(
            "Test image directory not found. Checked: "
            f"{requested_test_dir} and {existing_test_dir}"
        )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        path
        for path in test_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not image_files:
        raise FileNotFoundError(f"No test images found in: {test_images_dir}")

    print(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Running prediction on {len(image_files)} images in {test_images_dir}")
    results = model.predict(
        source=str(test_images_dir),
        imgsz=1280,
        conf=0.1,
        verbose=False,
    )

    predictions = []
    for result in results:
        image_name = Path(result.path).name
        boxes = result.boxes
        if boxes is None:
            print(f"Processed {image_name}: no detections")
            continue

        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for (x1, y1, x2, y2), class_value, confidence in zip(
            coordinates, class_ids, confidences
        ):
            class_id = int(class_value)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            predictions.append(
                {
                    "image_name": image_name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, "unknown"),
                    "confidence": float(confidence),
                    "bbox_x": x1,
                    "bbox_y": y1,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "bbox_area": bbox_w * bbox_h,
                    "bbox_height_px": bbox_h,
                }
            )
        print(f"Processed {image_name}: {len(coordinates)} detections")

    predictions_df = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
