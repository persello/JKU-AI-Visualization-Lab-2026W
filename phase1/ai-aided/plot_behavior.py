from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CLASS_ORDER = ["blue", "yellow", "orange", "large orange", "unknown"]
CLASS_COLORS = {
    "blue": "#1f77b4",
    "yellow": "#e6b800",
    "orange": "#ff7f0e",
    "large orange": "#d62728",
    "unknown": "#7f7f7f",
}
SIZE_ORDER = ["Small (<30 px)", "Medium (30-80 px)", "Large (>80 px)"]
REQUIRED_COLUMNS = {
    "image_name",
    "class_name",
    "confidence",
    "bbox_height_px",
}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "predictions.csv"
    output_dir = script_dir / "visualizations"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    predictions = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Predictions CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if predictions.empty:
        raise ValueError(f"Predictions CSV contains no detections: {csv_path}")

    predictions["confidence"] = pd.to_numeric(
        predictions["confidence"], errors="coerce"
    )
    predictions["bbox_height_px"] = pd.to_numeric(
        predictions["bbox_height_px"], errors="coerce"
    )
    if predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "Predictions CSV contains missing or invalid values in required columns."
        )

    predictions["class_name"] = predictions["class_name"].astype(str)
    predictions["size_category"] = np.select(
        [
            predictions["bbox_height_px"] < 30,
            predictions["bbox_height_px"] <= 80,
            predictions["bbox_height_px"] > 80,
        ],
        SIZE_ORDER,
        default=SIZE_ORDER[-1],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scatter = px.scatter(
        predictions,
        x="bbox_height_px",
        y="confidence",
        color="class_name",
        color_discrete_map=CLASS_COLORS,
        category_orders={"class_name": CLASS_ORDER},
        marginal_y="histogram",
        hover_data={"image_name": True},
        labels={
            "bbox_height_px": "Bounding Box Height (px)",
            "confidence": "Confidence",
            "class_name": "Cone Class",
            "image_name": "Image",
        },
        title="Confidence Decay vs. Cone Size",
        template="plotly_white",
    )

    for class_name in CLASS_ORDER:
        class_predictions = predictions[predictions["class_name"] == class_name]
        if len(class_predictions) < 2:
            continue
        x_values = class_predictions["bbox_height_px"].to_numpy()
        if np.unique(x_values).size < 2:
            continue
        slope, intercept = np.polyfit(
            x_values, class_predictions["confidence"].to_numpy(), 1
        )
        line_x = np.linspace(x_values.min(), x_values.max(), 100)
        scatter.add_trace(
            go.Scatter(
                x=line_x,
                y=slope * line_x + intercept,
                mode="lines",
                name=f"{class_name} trend",
                line={"color": CLASS_COLORS[class_name], "width": 2},
                legendgroup=class_name,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    distribution = px.box(
        predictions,
        x="class_name",
        y="confidence",
        color="class_name",
        color_discrete_map=CLASS_COLORS,
        category_orders={"class_name": CLASS_ORDER},
        points="outliers",
        labels={"class_name": "Cone Class", "confidence": "Confidence"},
        title="Confidence Distribution by Class",
        template="plotly_white",
    )

    size_distribution = px.histogram(
        predictions,
        x="size_category",
        color="class_name",
        color_discrete_map=CLASS_COLORS,
        category_orders={
            "size_category": SIZE_ORDER,
            "class_name": CLASS_ORDER,
        },
        barmode="group",
        labels={
            "size_category": "Bounding Box Height",
            "count": "Detections",
            "class_name": "Cone Class",
        },
        title="Cone Bounding Box Size Distribution",
        template="plotly_white",
    )

    charts = [
        (scatter, "confidence_decay_vs_cone_size.html"),
        (distribution, "confidence_distribution_by_class.html"),
        (size_distribution, "cone_bounding_box_size_distribution.html"),
    ]
    for figure, filename in charts:
        output_path = output_dir / filename
        figure.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved {output_path}")

    class_counts = (
        predictions["class_name"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )
    size_counts = (
        predictions.groupby(["class_name", "size_category"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=CLASS_ORDER, columns=SIZE_ORDER, fill_value=0)
    )
    print(f"\nTotal detections: {len(predictions):,}")
    print(f"Unique images: {predictions['image_name'].nunique():,}")
    print("\nDetections by class:")
    print(class_counts.to_string())
    print("\nDetections by bounding box size and class:")
    print(size_counts.to_string())
    print(
        "\nMean confidence: "
        f"{predictions['confidence'].mean():.3f} "
        f"(median: {predictions['confidence'].median():.3f})"
    )


if __name__ == "__main__":
    main()
