import math
from pathlib import Path

import cv2
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
    "frame_index",
    "timestamp_sec",
    "class_id",
    "class_name",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_height_px",
    "dist_from_center_px",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "yolo26s-fsoco5.pt"
    video_path = project_root / "data" / "videos" / "fs_east.mp4"
    output_path = Path(__file__).resolve().parent / "video_predictions.csv"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video file: {video_path}")

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not math.isfinite(width) or width <= 0 or not math.isfinite(height) or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid frame dimensions ({width}x{height}): {video_path}")
    frame_center_x = width / 2.0
    frame_center_y = height / 2.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video has an invalid frame rate ({fps}): {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({total_frames} frames at {fps:.3f} FPS)")
    predictions = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(
                source=frame,
                imgsz=1280,
                conf=0.1,
                verbose=False,
            )[0]
            boxes = result.boxes
            detection_count = 0

            if boxes is not None:
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for (x1, y1, x2, y2), class_value, confidence in zip(
                    coordinates, class_ids, confidences
                ):
                    class_id = int(class_value)
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    box_center_x = x1 + bbox_w / 2.0
                    box_center_y = y1 + bbox_h / 2.0

                    predictions.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": frame_index / fps,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES.get(class_id, "unknown"),
                            "confidence": float(confidence),
                            "bbox_x": x1,
                            "bbox_y": y1,
                            "bbox_w": bbox_w,
                            "bbox_h": bbox_h,
                            "bbox_height_px": bbox_h,
                            "dist_from_center_px": math.hypot(
                                box_center_x - frame_center_x,
                                box_center_y - frame_center_y,
                            ),
                        }
                    )
                detection_count = len(coordinates)

            frame_number = frame_index + 1
            if total_frames > 0:
                print(
                    f"Processed frame {frame_number}/{total_frames}: "
                    f"{detection_count} detections"
                )
            else:
                print(f"Processed frame {frame_number}: {detection_count} detections")
            frame_index += 1
    finally:
        capture.release()

    predictions_df = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved {len(predictions_df)} detections from {frame_index} frames to {output_path}")


if __name__ == "__main__":
    main()
