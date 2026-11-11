# Reflection on GenAI use in Phase I

AI-aided pair (Phase I):

- Eram Ansari, k12315344
- Arteljana Saliasi, k12542264

## What we used GenAI for

GitHub Copilot wrote all the code in `phase1/ai-aided/`: inference on the test split (962 images) and on an onboard video (FS East), and the Plotly charts of confidence against cone size, class, distance from the image centre, and still images against video. We chose the questions, the settings (`imgsz=1280`, `conf=0.1`) and the charts. The prompts are in [`ai-interactions/phase1/ai-interactions.md`](ai-interactions/phase1/ai-interactions.md). We also used Claude Code to turn the charts into images for the slides and to draft the slide text and this reflection, which we checked and edited.

## What worked

Each script worked within one or two prompts, much faster than writing it ourselves. Detailed prompts that listed every column, path and parameter gave the best code. The interactive charts made it easy to zoom in on small cones.

## Where it failed

- The video script assumed a 640 × 640 image centre, so distances were wrong on 1080p video.
- OpenCV could not read the AV1 video; we re-downloaded it in H.264.
- The straight trend lines go above a confidence of 1 and do not fit a curve that levels off.
- Class colours differ between charts.

The code always ran without errors. The mistakes were in what it computed or showed, so we only found them by checking the output against what is possible.

## How sure we are

Confidence drops from a median of 0.73 on still images to about 0.53 on video, and unknown and large orange cones get the lowest confidence. We trust both directions, but the video result comes from one video. We looked only at confidence, not at the labels, so we cannot say how many cones were missed. The human-only pair matched every box to the labels, so their results rest on firmer evidence. Both pairs agree that small cones are the weak point and motion makes it worse.

## For Phase II

Without GenAI, we will compare detections with the labels from the start, choose each chart's encoding ourselves, and check every number against what is possible before reading a chart.
