// Final presentation, end of Phase II. Recorded, 10 minutes at most: going over costs points.
// The slide deck is submitted too. The course asks for one story: problem and motivation,
// methods and findings of both pairs for Phase I and then Phase II, the AI4Vis reflection,
// and a conclusion.
//
// Time budget (speaker notes repeat it per slide):
//   problem and model 1:00 · Phase I 3:00 · Phase II 3:00 · AI4Vis reflection 2:00 ·
//   conclusion 1:00 = 10:00
//
// From the repository root:
//   typst compile --root . --input mode=handout      slides/final.typ slides/final.pdf
//   typst compile --root . --input mode=presentation slides/final.typ slides/final-presentation.pdf

#import "template/slides-template.typ": *
#import "team.typ": *

#show: jku-theme.with(
  config-info(
    course: course,
    title: title,
    subtitle: [Final presentation],
    group: group,
    author: everyone,
    date: datetime(year: 2026, month: 12, day: 9),
    institution: institution,
  ),
)

#title-slide()

= Problem and motivation

== Cone detection

- Formula Student Driverless cars find the track from the cones on both sides.
#pause
- Our question: how does detection change with the cone's distance and motion?
#pause
- #todo[Why it matters]

#speaker-note[About 0:30.]

== Model and data

- YOLO26s, pretrained on COCO, fine-tuned on FSOCO: blue, yellow, orange, large orange and unknown cones.
#pause
- FSOCO split by team, so validation and test teams are never seen in training.
#pause
- #todo[Data used in the analysis]

#speaker-note[About 0:30.]

= Phase I: how does the model behave?

== Common direction

- Main questions: #todo[...]
#pause
- Data: #todo[...]
#pause
- First visualization: #todo[...]

#speaker-note[About 0:20.]

== Human-only pair

#names(pair-a), without GenAI.

#pause

- Method: #todo[...]
#pause
- Finding: #todo[...]

#speaker-note[About 0:30.]

== #todo[Visualization title]

#slot[Key visualization of the human-only pair]

#speaker-note[About 0:50.]

== AI-aided pair

#names(pair-b), with GenAI.

#pause

- Method: #todo[...]
#pause
- Finding: #todo[...]

#speaker-note[About 0:30.]

== #todo[Visualization title]

#slot[Key visualization of the AI-aided pair]

#speaker-note[About 0:50.]

= Phase II: what did the model learn?

== Common direction

- Phase I findings we explain: #todo[...]
#pause
- The pairs swap: #names(pair-b) without GenAI, #names(pair-a) with it.

#speaker-note[About 0:20.]

== Human-only pair

#names(pair-b), without GenAI.

#pause

- Techniques: #todo[technique 1] and #todo[technique 2]
#pause
- What they show together: #todo[...]

#speaker-note[About 0:30. Each pair uses at least two techniques and compares or combines them.]

== #todo[Visualization title]

#slot[Key visualization of the human-only pair]

#speaker-note[About 0:50.]

== AI-aided pair

#names(pair-a), with GenAI.

#pause

- Techniques: #todo[technique 1] and #todo[technique 2]
#pause
- What they show together: #todo[...]

#speaker-note[About 0:30.]

== #todo[Visualization title]

#slot[Key visualization of the AI-aided pair]

#speaker-note[About 0:50.]

= AI4Vis reflection

== What we found

- Found by only one pair: #todo[...]
#pause
- Visualization choices: #todo[...]

#speaker-note[About 0:30. Both phases.]

== How we worked

- Breadth and depth: #todo[...]
#pause
- Speed, and the time spent checking AI output: #todo[...]
#pause
- Confidence in the conclusions, and the evidence for it: #todo[...]

#speaker-note[About 0:30.]

== Where GenAI failed

- #todo[Failure pattern 1]
#pause
- #todo[Failure pattern 2]

#speaker-note[About 0:30. From the AI interaction records of both phases.]

== What worked

- #todo[Practice 1]
#pause
- #todo[Practice 2]

#speaker-note[About 0:30.]

= Conclusion

== What we learned

- About the model: #todo[...]
#pause
- About GenAI in the visualization pipeline: #todo[...]

#speaker-note[About 1:00.]
