// JKU slide theme for touying, adapted from the UE Probabilistic Models template
// (UE - Probabilistic Models/template/slides-template.typ): Helvetica, JKU-blue titles, blue
// square bullets, JKU wordmark left / group name centre / page number right in the footer,
// logo slide and upper-case title page with the JKU "K", thin progress bar, booktabs tables.
// Keep decks minimal: many slides, little on each, reveals with touying's #pause, #uncover
// and #only.
//
// Usage (paths are relative to the deck; compile with `--root` at the repository root):
//
//   #import "template/slides-template.typ": *
//   #show: jku-theme.with(config-info(title: [...], author: [...], group: [...], ...))
//   #logo-slide()
//   #title-slide()
//   = Section         // section slide
//   == Slide title    // starts a new slide
//   #speaker-note[...] // presenter notes
//   #todo[...]        // text still to write
//   #slot[...]        // space for a figure still to make
//
// Build both versions:
//   typst compile --root . --input mode=handout      deck.typ deck-handout.pdf
//   typst compile --root . --input mode=presentation deck.typ deck-presentation.pdf

#import "@preview/touying:0.6.1": *
#import "colors.typ": *

#let font-sans = ("Helvetica Neue", "Helvetica", "Arial")
#let font-mono = ("SF Mono", "Menlo", "DejaVu Sans Mono")

// ── Brand marks ─────────────────────────────────────────────────────────────

// The JKU "K": a bar under a chevron (coordinates from \jkuk in beamerthemejku.sty).
#let jku-k(height: 1cm, fill: jku-blue) = {
  let s = height / 3.35
  let pt(x, y) = (x * s, y * s)
  box(width: 3.7 * s, height: height, {
    place(polygon(fill: fill, pt(0, 2.55), pt(3.7, 2.55), pt(3.7, 3.35), pt(0, 3.35)))
    place(polygon(
      fill: fill,
      pt(1.85, 2.55),
      pt(3.7, 0.95),
      pt(3.7, 0),
      pt(1.85, 1.6),
      pt(0, 0),
      pt(0, 0.95),
    ))
  })
}

#let jku-logo(height: 1cm) = image("assets/jku_en.pdf", height: height)
#let jku-wordmark(height: 9pt) = image("assets/jkuonly.png", height: height)

// ── Building blocks ─────────────────────────────────────────────────────────

#let alert(body) = text(fill: jku-purple, weight: "bold", body)

#let cols(..columns, ratio: auto, gutter: 1.2em) = {
  let n = columns.pos().len()
  grid(
    columns: if ratio == auto { (1fr,) * n } else { ratio },
    column-gutter: gutter,
    align: top,
    ..columns
  )
}

// Text still to write. Grey and in brackets, so it is easy to spot in the PDF.
#let todo(body) = text(fill: jku-grey)[[#body]]

// Space for a figure still to make: a dashed frame that fills the rest of the slide (or
// `height`), with a short description in the middle. Replace it with the image.
#let slot(height: 1fr, body) = block(
  width: 100%,
  height: height,
  stroke: (paint: jku-rule, thickness: 1pt, dash: "dashed"),
  radius: 4pt,
  align(center + horizon, todo(body)),
)

// ── Slides ──────────────────────────────────────────────────────────────────

// Logo, title and section slides step this counter so that content slides are numbered
// 1..N like beamer frames. Touying's own counter still handles #pause subslides.
#let _uncounted = counter("jku-uncounted")
#let _frame() = utils.slide-counter.get().first() - _uncounted.get().first()
#let _frames() = utils.slide-counter.final().first() - _uncounted.final().first()

#let _progress = context {
  let current = _frame()
  let total = calc.max(_frames(), 1)
  place(top + left, rect(width: 100% * calc.min(current / total, 1), height: 2pt, fill: jku-blue))
}

#let slide(title: auto, ..args) = touying-slide-wrapper(self => {
  if title != auto { self.store.title = title }

  let header(self) = pad(x: self.page.margin.at("x"), {
    set align(left + bottom)
    set text(fill: jku-blue)
    text(size: 1.25em, weight: "bold", if self.store.title != none { self.store.title } else {
      utils.display-current-heading(level: 2, numbered: false)
    })
  })

  let footer(self) = pad(x: self.page.margin.at("x"), {
    set text(size: 10pt, fill: jku-grey)
    line(length: 100%, stroke: 0.5pt + jku-rule)
    v(-0.2em)
    grid(
      columns: (1fr, auto, 1fr),
      align: (left + horizon, center + horizon, right + horizon),
      jku-wordmark(height: 9pt),
      self.info.at("group", default: none),
      context [#_frame()/#_frames()],
    )
  })

  self = utils.merge-dicts(self, config-page(header: header, footer: footer, background: _progress))
  touying-slide(self: self, ..args)
})

// Plain slide with only the JKU logo. Not counted in the page numbers.
#let logo-slide() = touying-slide-wrapper(self => {
  self = utils.merge-dicts(self, config-page(header: none, footer: none, margin: 0pt))
  touying-slide(self: self, {
    _uncounted.step()
    align(center + horizon, jku-logo(height: 3.6cm))
  })
})

// Title page. Not counted in the page numbers.
#let title-slide() = touying-slide-wrapper(self => {
  let info = self.info
  self = utils.merge-dicts(self, config-page(header: none, footer: none))
  let body = {
    _uncounted.step()
    set align(left + horizon)
    block(width: 80%, {
      if info.at("course", default: none) != none {
        text(size: 0.8em, fill: jku-grey, info.course)
        v(0.5em)
      }
      par(leading: 0.28em, text(size: 1.75em, weight: "bold", fill: jku-blue, upper(info.title)))
      if info.subtitle != none {
        v(0.35em)
        text(size: 1em, fill: jku-dark-grey, info.subtitle)
      }
      v(0.7em)
      jku-k(height: 1.15cm)
      v(0.7em)
      set text(size: 0.8em)
      if info.at("group", default: none) != none { text(weight: "bold", info.group) }
      if info.author != none {
        linebreak()
        info.author
      }
      let meta = (utils.display-info-date(self), info.institution).filter(x => x != none and x != [] and x != "")
      if meta.len() > 0 {
        linebreak()
        text(fill: jku-dark-grey, meta.join[ #sym.dot.c ])
      }
    })
    place(bottom + left, jku-logo(height: 1.15cm))
  }
  touying-slide(self: self, body)
})

#let section-slide(..args) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(self, config-page(header: none, footer: none, background: _progress))
  touying-slide(self: self, {
    _uncounted.step()
    set align(left + horizon)
    text(size: 1.8em, weight: "bold", fill: jku-blue, upper(utils.display-current-heading(
      level: 1,
      numbered: false,
    )))
    v(0.6em)
    jku-k(height: 1.15cm)
  })
})

#let jku-theme(
  aspect-ratio: "16-9",
  section-slides: true,
  size: 18pt,
  // "handout": reveals collapsed, no notes (the PDF to submit).
  // "presentation": reveals as separate pages, #speaker-note shown on a second screen.
  mode: sys.inputs.at("mode", default: "handout"),
  ..args,
  body,
) = {
  set text(size: size, font: font-sans, fill: jku-text, lang: "en")
  set par(leading: 0.55em, spacing: 0.8em, justify: false)
  set list(
    marker: (
      text(fill: jku-blue, size: 0.75em, baseline: -0.08em, sym.square.filled),
      text(fill: jku-blue, size: 0.75em, baseline: -0.08em, sym.square.stroked),
      text(fill: jku-blue, sym.bullet),
    ),
    spacing: 0.75em,
    body-indent: 0.55em,
  )
  set enum(numbering: n => text(fill: jku-blue, weight: "bold", numbering("1.", n)), spacing: 0.75em)
  show strong: set text(weight: "bold")
  show raw: set text(font: font-mono)
  show raw.where(block: false): set text(size: 1.05em)
  show raw.where(block: true): it => block(
    width: 100%,
    fill: jku-tint,
    inset: 0.6em,
    radius: 4pt,
    text(size: 0.8em, it),
  )
  // Booktabs-style tables.
  set table(
    stroke: (x, y) => (
      top: if y == 0 { 1pt + jku-blue } else { 0pt },
      bottom: if y == 0 { 0.6pt + jku-blue } else { 0.4pt + jku-rule },
    ),
    fill: (x, y) => if y > 0 and calc.even(y) { jku-tint },
    inset: (x: 0.55em, y: 0.38em),
  )
  show table.cell.where(y: 0): set text(weight: "bold", fill: jku-blue)

  show: touying-slides.with(
    config-page(
      paper: "presentation-" + aspect-ratio,
      margin: (x: 1.6cm, top: 2.7cm, bottom: 1.5cm),
      header-ascent: 0.9em,
      footer-descent: 0.6em,
    ),
    config-common(
      handout: mode == "handout",
      show-notes-on-second-screen: if mode == "presentation" { right } else { none },
      slide-fn: slide,
      new-section-slide-fn: if section-slides { section-slide } else { none },
    ),
    config-colors(primary: jku-blue, secondary: jku-dark-grey, tertiary: jku-grey, neutral: jku-text),
    config-store(title: none),
    ..args,
  )

  body
}
