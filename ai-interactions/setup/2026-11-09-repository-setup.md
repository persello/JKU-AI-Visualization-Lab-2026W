# 2026-11-09: repository and training setup

Who: Riccardo Persello
Tool and model: Claude Code (Claude Opus 5.5)

## Important prompts

- Prepare the repository for the whole group, for both pairs.
- Move our 2024 cone detector (YOLOv8n trained on FSOCO, Kaggle model `persello/colo`) to current tools and YOLO26.

## What the AI did

- Read the 2024 checkpoint's training arguments as data: YOLOv8n, Ultralytics 8.1.5, imgsz 1408, 100 epochs (best at 86), 10 classes.
- Set up the uv environment (Ultralytics 8.4, PyTorch, ruff, ty).
- Wrote `scripts/prepare_data.py`, which merges the 10 FSOCO classes into 5 (the `seg_` classes are the same cones from the segmentation part of FSOCO) and turns polygon labels into boxes.
- Wrote `scripts/train.py` for fine-tuning YOLO26.
- Created the folder layout and the files the course asks for (`team-info.json`, `workload.json`, this folder).

## What we accepted, changed or rejected, and why

Setup only. No Phase I analysis, visualization or interpretation was done with AI here, so the human-only pair starts from the same point as the AI-aided pair.
