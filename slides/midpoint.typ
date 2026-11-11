// Midpoint progress video, end of Phase I. At most 5 minutes: going over costs points.
// The course asks for: a short introduction to the problem, what each pair did and found,
// and the key visualizations.
//
// Time budget (speaker notes repeat it per slide):
//   problem, model and data 0:40 · common direction 0:20 · human-only pair 1:30 ·
//   AI-aided pair 1:30 · comparison 0:45 · next steps 0:15 = 5:00
//
// From the repository root:
//   typst compile --root . --input mode=handout      slides/midpoint.typ slides/midpoint.pdf
//   typst compile --root . --input mode=presentation slides/midpoint.typ slides/midpoint-presentation.pdf

#import "template/slides-template.typ": *
#import "team.typ": *

#show: jku-theme.with(
  config-info(
    course: course,
    title: title,
    subtitle: [Midpoint update. Phase I: how does the model behave?],
    group: group,
    author: everyone,
    date: datetime(year: 2026, month: 11, day: 11),
    institution: institution,
  ),
)

#title-slide()

= Problem

== Cone detection

- Formula Student Driverless cars find the track from the cones on both sides.
#pause
- Our question: how does detection change with the cone's distance and motion?
#pause
- Far cones are only a few pixels tall and a moving car blurs them, exactly where the car must look ahead.

#speaker-note[About 0:20.]

== Model and data

- YOLO26s, pretrained on COCO, fine-tuned on FSOCO: blue, yellow, orange, large orange and unknown cones.
#pause
- FSOCO split by team, so validation and test teams are never seen in training.
#pause
- Phase I: the test split (teams never seen in training) and onboard videos from Formula Student events.

#speaker-note[About 0:20.]

== Common direction

- Main questions: how do recall and confidence change with cone size (distance) and with motion?
#pause
- Data: FSOCO test split and onboard driverless videos.
#pause
- First visualization: detection quality against cone height in pixels.

#speaker-note[About 0:20. Agreed by the whole group before the pairs split.]

= Human-only pair

== What we did

#names(pair-a), without GenAI.

#pause

- Matched every detection to the labels and measured recall and confidence by cone height.
#pause
- Added synthetic motion blur (2 directions, 4 lengths) and tracked cones in 3 onboard videos.

#speaker-note[About 0:20.]

== Recall by cone size

#align(center, image("../phase1/human-only/figures/recall_by_size.svg", height: 85%))

#speaker-note[About 0:30.]

== Small cones are the weak point

- Recall above 99 % for cones taller than 23 px, but 77 % below 6 px, where precision is only 42 %.
#pause
- 19,089 labelled cones from teams never seen in training; the drop shows in every class.
#pause
- A cone a few pixels tall shows almost no shape or colour for the model to use.

#speaker-note[About 0:20.]

== Blur hurts small cones first

- Small cones (< 11 px): recall 85 % → 32 % at 8 px of horizontal blur, almost 0 % at 16 px.
#pause
- Same images, only the blur changes; large cones keep 99 % recall even at 16 px.
#pause
- Blur spreads a small cone's few pixels until its outline disappears.

#speaker-note[About 0:20.]

= AI-aided pair

== What we did

#names(pair-b), with GenAI.

#pause

- Ran the model on the static test split and an onboard video (FS East), saving confidence, box size and distance from the image centre.
#pause
- Plotted confidence against cone size, class, distance from the centre, and still images vs. video.
#pause
- GitHub Copilot wrote the inference and Plotly scripts from our prompts (record in `ai-interactions/`)

#speaker-note[About 0:20.]

== Confidence vs. cone size

#align(center, image("../phase1/ai-aided/figures/confidence_decay_vs_cone_size.png", height: 85%))

#speaker-note[About 0:30.]

== Cone sizes in the test set

#align(center, image("../phase1/ai-aided/figures/cone_bounding_box_size_distribution.png", height: 85%))

#speaker-note[Extra figure: cut if over time.]

== Confidence by class

#align(center, image("../phase1/ai-aided/figures/confidence_distribution_by_class.png", height: 85%))

#speaker-note[Extra figure: cut if over time.]

== Lens distortion and confidence

#align(center, image("../phase1/ai-aided/figures/distortion_vs_confidence.png", height: 85%))

#speaker-note[Extra figure: cut if over time.]

== Still images vs. video frames

#align(center, image("../phase1/ai-aided/figures/static_vs_video_comparison.png", height: 85%))

#speaker-note[Extra figure: cut if over time.]

== Confidence drops in video

- Median confidence 0.73 on still test images, about 0.53 on video frames.
#pause
- Same model and settings (`imgsz=1280`, `conf=0.1`) on thousands of detections from both sources.
#pause
- Motion blur, and a different camera and lighting than in the test images.

#speaker-note[About 0:20.]

== Unknown and large orange cones are least certain

- Median confidence about 0.22 for unknown and 0.62 for large orange cones, 0.73–0.75 for the others.
#pause
- Holds over all test detections; most detected cones are small (< 30 px).
#pause
- These classes are rare in FSOCO, so the model saw few examples.

#speaker-note[About 0:20.]

= Comparison

== What we found

- Found by only one pair: blur and recall against labels (human-only); the confidence drop in video and the distortion view (AI-aided).
#pause
- Visualization choices: static figures with confidence intervals (human-only); interactive Plotly charts of confidence (AI-aided).

#speaker-note[About 0:20.]

== How we worked

- Breadth and depth: the AI-aided pair covered more chart types quickly; the human-only pair went deeper, checking every box against the labels.
#pause
- Speed, and the time spent checking AI output: Copilot wrote working scripts fast, but a hard-coded image centre and a video codec problem had to be found and fixed.
#pause
- Confidence in the conclusions, and the evidence for it: higher for the human-only results, which use labels; the AI-aided charts show confidence only, and their trend lines go above 1.

#speaker-note[About 0:25.]

== Next: Phase II

- The pairs swap: #names(pair-b) without GenAI, #names(pair-a) with it.
#pause
- Findings to explain: why small and blurred cones are missed, and why confidence drops in video.

#speaker-note[About 0:15.]
