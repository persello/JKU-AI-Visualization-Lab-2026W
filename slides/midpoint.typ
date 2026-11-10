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
- #todo[What makes this interesting or hard, in one line]

#speaker-note[About 0:20.]

== Model and data

- YOLO26s, pretrained on COCO, fine-tuned on FSOCO: blue, yellow, orange, large orange and unknown cones.
#pause
- FSOCO split by team, so validation and test teams are never seen in training.
#pause
- #todo[Which images we run the model on for Phase I]

#speaker-note[About 0:20.]

== Common direction

- Main questions: #todo[...]
#pause
- Data: #todo[...]
#pause
- First visualization: #todo[...]

#speaker-note[About 0:20. Agreed by the whole group before the pairs split.]

= Human-only pair

== What we did

#names(pair-a), without GenAI.

#pause

- #todo[Approach, step 1]
#pause
- #todo[Approach, step 2]

#speaker-note[About 0:20.]

== #todo[Visualization title]

#slot[Key visualization]

#speaker-note[About 0:30.]

== #todo[Finding 1]

- #todo[The pattern]
#pause
- #todo[The evidence that it is systematic]
#pause
- #todo[Why it happens, as far as we know]

#speaker-note[About 0:20.]

== #todo[Finding 2]

- #todo[The pattern]
#pause
- #todo[The evidence that it is systematic]
#pause
- #todo[Why it happens, as far as we know]

#speaker-note[About 0:20.]

= AI-aided pair

== What we did

#names(pair-b), with GenAI.

#pause

- #todo[Approach, step 1]
#pause
- #todo[Approach, step 2]
#pause
- #todo[What GenAI was used for] (record in `ai-interactions/`)

#speaker-note[About 0:20.]

== #todo[Visualization title]

#slot[Key visualization]

#speaker-note[About 0:30.]

== #todo[Finding 1]

- #todo[The pattern]
#pause
- #todo[The evidence that it is systematic]
#pause
- #todo[Why it happens, as far as we know]

#speaker-note[About 0:20.]

== #todo[Finding 2]

- #todo[The pattern]
#pause
- #todo[The evidence that it is systematic]
#pause
- #todo[Why it happens, as far as we know]

#speaker-note[About 0:20.]

= Comparison

== What we found

- Found by only one pair: #todo[...]
#pause
- Visualization choices: #todo[...]

#speaker-note[About 0:20.]

== How we worked

- Breadth and depth: #todo[...]
#pause
- Speed, and the time spent checking AI output: #todo[...]
#pause
- Confidence in the conclusions, and the evidence for it: #todo[...]

#speaker-note[About 0:25.]

== Next: Phase II

- The pairs swap: #names(pair-b) without GenAI, #names(pair-a) with it.
#pause
- Findings to explain: #todo[...]

#speaker-note[About 0:15.]
