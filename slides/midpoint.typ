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
- The cones that matter most for planning are the far ones, and those are a few pixels tall.

#speaker-note[About 0:20.]

== Model and data

- YOLO26s, pretrained on COCO, fine-tuned on FSOCO: blue, yellow, orange, large orange and unknown cones.
#pause
- FSOCO split by team, so validation and test teams are never seen in training.
#pause
- Phase I: the test split, 962 images and 19,089 cones from 4 unseen teams, plus three onboard videos.

#speaker-note[About 0:20.]

== Common direction

- Main questions: how does detection change with a cone's distance, and with motion?
#pause
- Data: the FSOCO test split, blurred copies of it, and three onboard race videos.
#pause
- First visualization: recall against the cone's height in the image.

#speaker-note[About 0:20. Agreed by the whole group before the pairs split.]

= Human-only pair

== What we did

#names(pair-a), without GenAI.

#pause

- Every box on the test split, down to confidence 0.001, matched to the labels at IoU 0.5.
#pause
- Cones binned by their height at the model input: height stands for distance.
#pause
- Motion, which FSOCO does not label: blurred copies of the same images, and the onboard videos.

#speaker-note[About 0:20. Four teams the model has never seen. We keep every box, so the
threshold stays ours to choose afterwards.]

== Recall falls below 16 px, fastest for orange

#block(
  height: 1fr,
  width: 100%,
  image("../phase1/human-only/figures/recall_by_size.svg", width: 100%, height: 100%, fit: "contain"),
)

#speaker-note[About 0:30. Cone height at the model input on the x axis, log scale. Watch where
each colour leaves the plateau: large orange goes first, at about 16 px.]

== Distance: small cones are missed, not mistaken

- Recall 0.99 above 23 px, 0.77 under 6 px. Precision falls further: 0.97 to 0.42.
#pause
- Systematic, not cherry-picked: 19,089 cones, and 85 % of the false boxes are under 11 px.
#pause
- They are missed outright (15 %), not given the wrong class (1 %).
#pause
- But #alert[the camera matters as much as the size]: in the smallest bin, 0.86 for aristurtle
  against 0.12 for amz.

#speaker-note[About 0:20. Median confidence stays above the 0.25 threshold even for the smallest
cones, so the threshold is not what removes them: the evidence thins.]

== Motion: a cone survives until the blur is its own width

- Blur about as long as a cone is wide removes it, and small cones go first.
#pause
- Under 11 px: recall 0.85 unblurred, 0.75 at 4 px, 0.32 at 8 px, 0.005 at 16 px.
#pause
- Sideways costs more than vertical, 0.32 against 0.50 at 8 px: a cone is narrower than it is tall.
#pause
- On real video the detector keeps up, #alert[the tracker does not]: BoT-SORT holds 18–33 % of the
  detections while driving, against about half while the car stands.

#speaker-note[About 0:20. The blur is synthetic because FSOCO has no motion labels. The last bullet
is the surprise: the weak part is outside the model, in the tracker's default settings.]

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
