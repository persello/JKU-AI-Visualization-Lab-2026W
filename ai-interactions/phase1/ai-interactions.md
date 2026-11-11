# 2026-11-10: Phase I Static Baseline Inference & Bounding Box Behavior

Who: AI-aided pair (Phase I)  

Tool and model: GitHub Copilot (Auto mode)

## Important prompts

    "Act as a PyTorch and Computer Vision expert. Write a Python script saved to phase1/ai-aided/run_inference.py to evaluate our fine-tuned YOLO model on the static test split.

    Requirements:

        Load model weights from yolo26s-fsoco5.pt.

        Run inference on all images in data/fsoco5/images/test using parameters imgsz=1280 and conf=0.1.

        Class ID mapping: {0: 'blue', 1: 'yellow', 2: 'orange', 3: 'large orange', 4: 'unknown'}.

        For every detection, record: image_name, class_id, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h, bbox_height_px, bbox_area_px.

        Save all detection records into a Pandas DataFrame and export to phase1/ai-aided/predictions.csv. Add console progress logging."

## What the AI suggested

Copilot generated run_inference.py using ultralytics.YOLO to iterate through the test set, extract bounding box metrics, map class names, and export the structured dataset to predictions.csv.

## What we accepted, changed or rejected, and why

    Accepted: The frame iteration loop and CSV output structure.

    Changed: Explicitly set confidence threshold to conf=0.1 and image size to imgsz=1280 to match evaluation specifications.

## Failure patterns noticed

None. The static batch inference pipeline executed without issues on the 962 test set images.

## What worked well

Automating the batch extraction of bounding box heights and confidence scores into a structured CSV saved manually writing coordinate parsing loops.
2026-11-10: Phase I Static Baseline Visualization

Who: AI-aided pair (Phase I) 

Tool and model: GitHub Copilot (Auto mode)

## Important prompts

    "Act as a Data Visualization expert. Write a Python script saved to phase1/ai-aided/plot_behavior.py that reads phase1/ai-aided/predictions.csv.

    Requirements:

        Load predictions.csv using Pandas.

        Create output directory phase1/ai-aided/visualizations/ if it does not exist.

        Generate Confidence Decay vs. Visual Size (confidence_decay.html): A Plotly scatter plot (plotly.express.scatter) with bbox_height_px (X-axis) vs. confidence (Y-axis), color-coded by class_name. Add ordinary least squares trendlines (trendline='ols') to analyze how prediction certainty scales with visual object size.

        Save the interactive HTML dashboard to phase1/ai-aided/visualizations/confidence_decay.html and print summary statistics to the console."

## What the AI suggested

Copilot created a Plotly visualizer script that generates interactive HTML charts showing confidence scaling against cone pixel height.

## What we accepted, changed or rejected, and why

    Accepted: Plotly scatter plot with OLS trendlines.

    Changed: Configured output path specifically to phase1/ai-aided/visualizations/confidence_decay.html.

## Failure patterns noticed

Ensured statsmodels package was present in the virtual environment for OLS trendline calculation.

## What worked well

Interactive Plotly HTML format allowed easy filtering by cone class and visual inspection of small-object confidence degradation.
2026-11-10: Dynamic Video Inference & Radial Distance Calculation

Who: AI-aided pair (Phase I) 

Tool and model: GitHub Copilot (Auto mode)

## Important prompts

    "Act as a PyTorch and OpenCV expert. Write a Python script saved to phase1/ai-aided/run_video_inference.py that processes a video file using our yolo26s-fsoco5.pt model.

    Requirements:

        Use OpenCV (cv2.VideoCapture) to open data/videos/fs_east.mp4 and load weights from yolo26s-fsoco5.pt.

        Dynamically extract frame width W and height H from cv2.VideoCapture properties to compute the exact image center: center_x = W / 2.0 and center_y = H / 2.0.

        Pass parameters imgsz=1280 and conf=0.1 to the model.

        Class ID mapping: {0: 'blue', 1: 'yellow', 2: 'orange', 3: 'large orange', 4: 'unknown'}.

        For every detected cone in every frame, extract:

            frame_index

            timestamp_sec (calculated as frame_index / fps)

            class_id and class_name

            confidence

            Bounding box coordinates: bbox_x, bbox_y, bbox_w, bbox_h, bbox_height_px

            Bounding box center: box_center_x = bbox_x + bbox_w / 2.0 and box_center_y = bbox_y + bbox_h / 2.0

            Radial distance from image center: dist_from_center_px = sqrt((box_center_x - center_x)^2 + (box_center_y - center_y)^2)

        Save all detection records into a DataFrame and overwrite phase1/ai-aided/video_predictions.csv. Add console output to track frame processing progress."

## What the AI suggested

Copilot provided an OpenCV frame-by-frame inference loop that dynamically queries video properties to compute distance from center per bounding box and exports data to video_predictions.csv.

## What we accepted, changed or rejected, and why

    Accepted: Dynamic center computation (W/2, H/2) directly from OpenCV context.

    Rejected: Previous hardcoded (640, 640) center offset prompt, which produced erroneous radial distances exceeding frame boundaries for 1080p video.

## Failure patterns noticed

    Codec Incompatibility: Default video stream using AV1 codec failed in OpenCV frame grab ([av1] Failed to get pixel format). Resolved by re-downloading standard H.264 (AVC) encoding via yt-dlp.

    Hardcoded Image Center: Static center assumptions distorted maximum radial distance math on 1920x1080 resolution.


## What worked well

Dynamic resolution fetching from OpenCV properties made the radial calculation robust across any video aspect ratio.
2026-11-10: Dynamic Video vs. Static Performance & Lens Distortion Dashboard

Who: AI-aided pair (Phase I) 

Tool and model: GitHub Copilot (Auto mode)

## Important prompts

    "Act as a Data Visualization expert. Write a Python script saved to phase1/ai-aided/plot_video_behavior.py that reads phase1/ai-aided/video_predictions.csv and phase1/ai-aided/predictions.csv.

    Requirements:

        Load both CSV datasets using Pandas.

        Ensure output directory phase1/ai-aided/visualizations/ exists.

        Generate Lens Distortion Impact (distortion_vs_confidence.html): A Plotly scatter plot (plotly.express.scatter) with Distance from Image Center (dist_from_center_px) on the X-axis and confidence on the Y-axis, color-coded by class_name. Add ordinary least squares trendlines (trendline='ols') to evaluate confidence behavior across spatial distances.

        Generate Static Set vs. Dynamic Video Confidence (static_vs_video_comparison.html): A side-by-side box plot (plotly.express.box) comparing confidence score distributions between static test images (predictions.csv) and dynamic video frames (video_predictions.csv).

        Save both HTML dashboards to phase1/ai-aided/visualizations/ and log summary statistics in the console."

## What the AI suggested

Copilot generated a comparative visualization script outputting side-by-side static vs. video box plots and a radial distance scatter plot.

## What we accepted, changed or rejected, and why

    Accepted: Box plots comparing median confidence shift between static and dynamic video datasets.

    Accepted: Scatter plot showing confidence ceiling roll-off near perimeter image bounds (~1102 px).

## Failure patterns noticed

Browsers aggressively cache HTML/JS files; hard-refresh (Ctrl+F5) was needed to confirm updated Plotly data ranges.

## What worked well

Combining static test metrics and frame-by-frame video metrics into a single comparison script clearly demonstrated the ~0.21 drop in median detection confidence during dynamic driving sequences.