# Phase I: how does the model behave?, ai-aided pair

Pair: 

Generative AI is part of the process. Record the important interactions in `ai-interactions/phase1/`.

Run video inference on `data/videos/fs_east.mp4` with `uv run python phase1/ai-aided/run_video_inference.py`.
The script loads `yolo26s-fsoco5.pt` and writes detections to `phase1/ai-aided/video_predictions.csv`.

Generate video behavior visualizations with `uv run python phase1/ai-aided/plot_video_behavior.py`.
The script compares video detections with `predictions.csv` and writes the HTML plots to `phase1/ai-aided/visualizations/`.
