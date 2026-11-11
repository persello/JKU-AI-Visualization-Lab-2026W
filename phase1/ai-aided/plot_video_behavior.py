from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

VIDEO_REQUIRED_COLUMNS = {"dist_from_center_px", "confidence", "class_name"}
STATIC_REQUIRED_COLUMNS = {"confidence"}


def load_predictions(csv_path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    predictions = pd.read_csv(csv_path)
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if predictions.empty:
        raise ValueError(f"Predictions CSV contains no detections: {csv_path}")

    for column in required_columns & {"confidence", "dist_from_center_px"}:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
        if predictions[column].isna().any() or not np.isfinite(
            predictions[column].to_numpy()
        ).all():
            raise ValueError(
                f"{csv_path.name} contains missing or invalid values in {column!r}."
            )

    if "class_name" in required_columns:
        if predictions["class_name"].isna().any():
            raise ValueError(f"{csv_path.name} contains missing class_name values.")
        predictions["class_name"] = predictions["class_name"].astype(str)

    return predictions


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "visualizations"
    video_predictions = load_predictions(
        script_dir / "video_predictions.csv", VIDEO_REQUIRED_COLUMNS
    )
    static_predictions = load_predictions(
        script_dir / "predictions.csv", STATIC_REQUIRED_COLUMNS
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    distortion_plot = px.scatter(
        video_predictions,
        x="dist_from_center_px",
        y="confidence",
        color="class_name",
        trendline="ols",
        labels={
            "dist_from_center_px": "Distance from Image Center (px)",
            "confidence": "Confidence",
            "class_name": "Class",
        },
        title="Lens Distortion Impact on Detection Confidence",
        template="plotly_white",
    )

    confidence_comparison = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "Static test images",
                    "confidence": static_predictions["confidence"],
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "Dynamic video frames",
                    "confidence": video_predictions["confidence"],
                }
            ),
        ],
        ignore_index=True,
    )
    comparison_plot = px.box(
        confidence_comparison,
        x="dataset",
        y="confidence",
        color="dataset",
        points="outliers",
        labels={"dataset": "Prediction source", "confidence": "Confidence"},
        title="Static Set vs. Dynamic Video Confidence",
        template="plotly_white",
    )

    charts = [
        (distortion_plot, "distortion_vs_confidence.html"),
        (comparison_plot, "static_vs_video_comparison.html"),
    ]
    for figure, filename in charts:
        output_path = output_dir / filename
        figure.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved {output_path}")

    print("\nVideo frame detection summary:")
    print(f"Detections: {len(video_predictions):,}")
    print("Confidence:")
    print(video_predictions["confidence"].describe().to_string())
    print("\nDistance from image center (px):")
    print(video_predictions["dist_from_center_px"].describe().to_string())

    print("\nStatic test image detection summary:")
    print(f"Detections: {len(static_predictions):,}")
    print("Confidence:")
    print(static_predictions["confidence"].describe().to_string())


if __name__ == "__main__":
    main()
