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

To train a new model, fine-tune YOLO26 (uses the GPU: CUDA or Apple MPS). Runs and weights go to `runs/`:

```bash
uv run scripts/train.py --model yolo26s.pt --imgsz 1280 --hours 8 --cache
```

Check code with `uv run ruff check` and `uv run ty check`.

Build the slides with [Typst](https://typst.app) from the repository root. `mode=presentation` shows each reveal step on its own page, with the speaker notes on a second screen:

```bash
typst compile --root . --input mode=handout slides/midpoint.typ slides/midpoint.pdf
```

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
