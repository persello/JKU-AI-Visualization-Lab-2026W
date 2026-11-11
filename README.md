# Azure Beetles: how a cone detector sees distance and motion

AI and Visualization lab, JKU, 2026W.

## Task, model and dataset

**Task.** Cone detection for Formula Student Driverless. We analyse how detection performance changes with distance and motion: the cone's visual size, and how much and in which direction it moves.

**Model.** YOLO26 ([Ultralytics](https://docs.ultralytics.com/models/yolo26/)), pretrained on COCO and fine-tuned on FSOCO with `scripts/train.py`. Our earlier model from 2024, a YOLOv8n trained from scratch on the same data, is on Kaggle as [`persello/colo`](https://www.kaggle.com/models/persello/colo); its code is in [`persello/daml-notebooks`](https://github.com/persello/daml-notebooks/tree/main/exam).

**Dataset.** [FSOCO](https://fsoco.github.io/fsoco-dataset/) (Formula Student Objects in Context), images from many Formula Student teams with cone annotations. We use our YOLO-format copy on Kaggle, [`persello/fsoco-yolo`](https://www.kaggle.com/datasets/persello/fsoco-yolo), with five classes: blue, yellow, orange, large orange and unknown cone.

`scripts/prepare_data.py` splits the images again by team: about 80 % for training, 10 % for validation and 10 % for testing, and every team is in only one split. The 2024 split was random per image (6534 / 1174 / 1215 images, all 41 teams in every split), so test images have near-copies (neighbouring frames) in the training set. `--split original` keeps it.

## Trained model

Our fine-tuned YOLO26s is in the release [`yolo26s-fsoco5-v1`](https://github.com/persello/JKU-AI-Visualization-Lab-2026W/releases/tag/yolo26s-fsoco5-v1) (19.5 MB):

```bash
curl -LO https://github.com/persello/JKU-AI-Visualization-Lab-2026W/releases/download/yolo26s-fsoco5-v1/yolo26s-fsoco5.pt
```

It was trained with the command at the end of Setup and stopped after epoch 7, when validation mAP50-95 reached a new best. On the validation split (1071 images from 3 teams that are not in the training split), precision is 0.727, recall 0.836, mAP50 0.834 and mAP50-95 0.543. Predict with `imgsz=1280`, the training image size.

## Setup

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Get the data. The download is a 19.2 GB zip that unpacks to about 20 GB. Unpacking it while it downloads needs only that much free space:

```bash
mkdir -p data/raw
curl -L "https://www.kaggle.com/api/v1/datasets/download/persello/fsoco-yolo?datasetVersionNumber=1" | bsdtar -xf - -C data/raw
```

At the end, curl may report `(56) Failure writing output`: bsdtar stops reading after the last file, before the zip's index. The download is complete if `data/raw/fsoco-yolo` has 8923 images (6534 in `train`, 1174 in `validate`, 1215 in `test`).

Then write the 5-class copy to `data/fsoco5/`, with symlinks to the images:

```bash
uv run scripts/prepare_data.py --source data/raw/fsoco-yolo
```

Without `--source`, the script downloads the dataset with kagglehub to `~/.cache/kagglehub` (set `KAGGLEHUB_CACHE` to change it), which needs about twice the space while it unpacks. On Linux, use `bsdtar` from the `libarchive-tools` package.

Check code with `uv run ruff check`, `uv run ruff format --check` and `uv run ty check`.

## Train

Fine-tune YOLO26 on the 5-class copy (uses the GPU: CUDA or Apple MPS). Runs and weights go to `runs/`:

```bash
uv run scripts/train.py --model yolo26s.pt --imgsz 1280 --hours 8 --cache
```

`--cache` keeps the resized images in RAM (about 20 GB). The best weights are written to `runs/yolo26s-1280/weights/best.pt`. To skip training, download our model from the release (see [Trained model](#trained-model)).

## Run inference

Use the Ultralytics command line with our weights (`runs/yolo26s-1280/weights/best.pt`, or `yolo26s-fsoco5.pt` from the release) and the training image size. `source` can be an image or a folder (for videos, see below). The annotated output goes to `runs/detect/predict*/`:

```bash
uv run yolo predict model=runs/yolo26s-1280/weights/best.pt source=data/fsoco5/test/images imgsz=1280
```

Onboard videos from Formula Student driverless events. They have no labels, but they show real motion:

| Video | Team |
|---|---|
| [Winning Driverless Trackdrive at FS Czech 2025](https://www.youtube.com/watch?v=o5vES5QaeiQ) | Chalmers Formula Student |
| [Formula Student East Driverless Autocross](https://www.youtube.com/watch?v=CrISno91wPg) | KA-RaceIng |
| [Skidpad Driverless ONBOARD at Formula Student Spain](https://www.youtube.com/watch?v=aPPk06QK_ME) | BCN eMotorsport |

The team `ka` in our training split is probably KA-RaceIng, so the model may know that car and camera. Chalmers and BCN eMotorsport do not appear by name in FSOCO.

Download a video with [yt-dlp](https://github.com/yt-dlp/yt-dlp) (video only, up to 1080p) to `data/videos/`. The three videos take about 300 MB. They belong to their teams: keep them out of git (`data/` is ignored).

```bash
uvx yt-dlp -f "bv*[height<=1080][ext=mp4]" -o "data/videos/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=o5vES5QaeiQ"
```

Then track the cones through the videos (Ultralytics' BoT-SORT, which gives each cone an ID across frames). The script runs in stream mode: without it, Ultralytics keeps the results of every frame in memory. It writes the annotated videos and a CSV of all tracked boxes to `runs/phase1/videos/`. The three videos take about 8 minutes on an Apple-silicon laptop:

```bash
uv run phase1/human-only/video_track.py data/videos/*.mp4
```

## Phase I analysis: cone size and motion

The results and which figure goes on which slide are in [`phase1/human-only/FINDINGS.md`](phase1/human-only/FINDINGS.md). Each script's docstring describes its method.

Run the model on the test split (4 teams not in training). It saves every box down to confidence 0.001 to `runs/phase1/test/`:

```bash
uv run phase1/human-only/predict_test.py
```

Match the boxes to the labels and draw the size figures and some example images. This takes a few seconds:

```bash
uv run phase1/human-only/size_analysis.py
```

```bash
uv run phase1/human-only/examples.py
```

For motion, run the model again on blurred copies of the test images (2 directions, 4 blur lengths, and no blur), then compare:

```bash
uv run phase1/human-only/blur_test.py
```

```bash
uv run phase1/human-only/blur_analysis.py
```

For real motion, use the onboard videos (see [Run inference](#run-inference) for downloading them and for `video_track.py`). Run the detector alone on 5 frames per second, then compare the detections and the tracked boxes with the image motion, and describe the tracks:

```bash
uv run phase1/human-only/video_detect.py data/videos/*.mp4
```

```bash
uv run phase1/human-only/video_motion.py
```

```bash
uv run phase1/human-only/video_stability.py
```

The figures (PNG and SVG) go to `phase1/human-only/figures/` and the tables to `phase1/human-only/results/`.

## Slides

Build the slides with [Typst](https://typst.app) from the repository root. `mode=presentation` shows each reveal step on its own page, with the speaker notes on a second screen:

```bash
typst compile --root . --input mode=handout slides/midpoint.typ slides/midpoint.pdf
```

```bash
typst compile --root . --input mode=presentation slides/midpoint.typ slides/midpoint-presentation.pdf
```

Figures from `phase1/` can be placed with `#image("../phase1/human-only/figures/recall_by_size.svg")`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Data preparation and training, shared by both pairs |
| `phase1/ai-aided/`, `phase1/human-only/` | Phase I work of each pair |
| `phase2/ai-aided/`, `phase2/human-only/` | Phase II work of each pair (the pairs swap roles) |
| `slides/` | Typst slides for the midpoint video and the final presentation; names go in `slides/team.typ` |
| `ai-interactions/` | Record of the AI-aided pair's use of generative AI |
| `ai_reflection_midpoint.md` | Reflection on GenAI use in Phase I |
| `team-info.json`, `workload.json` | Team details for the course |

Data, runs and weights are not in git.
