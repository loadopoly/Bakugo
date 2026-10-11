# Condition grading: corners, edges, surface

From 2.24.0 the grade estimate covers all four aspects a grader looks at.
Each aspect is measured from the card's pixels, or it is reported as not
assessed. Before 2.24 only centering was measured. Corners, edges and surface
started at 10 and lost up to 2 points each depending on how cleanly the outline
was found. That penalty was capped, so a soft live frame always came out at
exactly PSA 8.

## Modules

| Module | Role |
|---|---|
| `cardcenter/condition.py` | One view: corners, edges, surface readings, face (front/back), pose |
| `cardcenter/condition_evidence.py` | Pools views per face; next-best-view guidance |
| `cardcenter/grading.py` | Per-aspect grade distributions and their composition |
| `cardcenter/data/condition_standards.json` | Resolution floors, category boundaries, nominal corner radius |

## One view (`condition.analyse_view`)

The card is warped flat (canonical frame, up to 16 px/mm) with 2.5 mm of
background kept round it. The card's true scale comes from its outline, not
from `CenteringResult.px_per_mm`, which is the rectification scale and is
clipped at 6 px/mm. Resolution is checked per corner and per side, so the far
edge of a tilted card counts with its own lower resolution.

**Colour models.** The border is sampled from bands just inside each edge. The
background is sampled just outside and split into up to three colour clusters,
because a card on a black bag with one grey fold needs two background models.
A corner or side is not resolvable if a sizeable background cluster looks like
the border.

**White fibre.** A pixel counts as white fibre when its lightness is above the
border's. Loss of colour is required as well, but never used on its own. Phone
JPEGs store colour at half resolution, so the outer half-millimetre of any
coloured border over a dark background loses colour while its lightness stays
the same. On the Machamp field photo the colour-only test reported 90% wear on a
clean edge.

**Corners.** Two straight edges are fitted 6-9.7 mm from the corner, past any
plausible worn arc. The nominal 3.0 mm rounded corner is placed tangent to
both. Rays from the arc's centre find where card material stops:

- `loss_mm`: how far inside the nominal outline the material stops, taken at the 75th percentile of the rays.
- `whitening_mm`: how deep the white run goes inward from that point.

Fitting the edges locally matters. The outline that placed the frame can be a
few tenths of a millimetre off on a tilted phone frame, and judged against it
an intact corner read 0.67 mm of loss.

**Edges.** Normal profiles are taken every 0.5 mm, with colour models per 10 mm
stretch. Nicks are inward departures from a running median of the cut over
±3 mm. A straight-line fit would flag a card's gentle bow: the fresh Probopass
bows 0.14 mm along its top edge. White fibre is measured starting past the
visible side face. The side face's width comes from the camera pose, which is
self-calibrated from the card's rectangular outline because the nominal field
of view is often wrong. Positions where the border well inside the edge is also
white are glare. They are excluded rather than scored.

**Resolution and blur tiers (2.25.0).** Sharpness is the 20-80% width of the edge-spread function,
measured on the pixel's position along the background-to-border colour axis,
averaged across rays or positions so print and background texture cancel:
- **Fine tier:** at least 6.0 px/mm locally, blur under 0.45 mm (0.3 mm for whitening).
- **Coarse tier (shop mode):** at least 3.5 px/mm locally, blur under 0.8 mm. Used when viewing cards across counters or in display cases. Readings below fine tier get `coarse=True` with wider measurement uncertainty ($\sigma = \sqrt{(0.6/\mathrm{res})^2 + (0.4\times\mathrm{blur})^2 + 0.03^2}$). White fibre is suppressed when coarse because half-millimetre colour subsampling cannot reliably separate fibre from background bleed.
Features softer than 0.8 mm or below 3.5 px/mm are not read at all.

**Surface.** Scratches and dents show in the reflection of a light, not in the
print. Each view records two things:

- Which 3 mm cells sat under a highlight. Thin dark lines inside a highlight are recorded as candidates.
- What the card looks like where there was no highlight (the diffuse print).

A highlight full of structure is foil. On foil, scratches can't be told from the
pattern, and the surface is reported as such.

**Face.** The card's back is recognised by three things together: a saturated
blue field, a blue border ring, and a red disc at the centre. Blue artwork on a
front fails the ring test or the disc test.

## Many views (`condition_evidence.ConditionEvidence`)

- **Whitening** is estimated from the low end of the views (mean of two; 30th percentile from three views on). Glare and a tilted card's side face can only make a border look whiter, never less white.
- **Loss and nicks** are geometry, the same in every view. They are estimated by the median. The uncertainty is floored by the resolution.
- **Surface defects** count only when two views see a candidate at the same place and direction, and the diffuse print there shows no such line. Dark ink under a highlight is therefore not reported as a scratch. Surface is assessed once 60% of the face has been under a highlight in two or more views.
- **Faces are kept apart.** The back's top-left corner is the front's top-right. Each physical corner and side takes the worse of its two faces.
- **`next_action()` (phone-centric):** names the one change of phone position or zoom that adds the most missing evidence:
  - zoom in (+) on the card when corners/edges need 3.5 px/mm;
  - move the phone toward a soft end or corner of the card;
  - hold the phone steady when the edges are soft;
  - tilt the phone slowly so the light's reflection moves across the card;
  - observe the back if visible.

## Grades (`grading.predict_overall_grade`)

Each aspect is a probability distribution over grades:

- **Centering** comes from the ratio's uncertainty and the spread between the strict and lenient published thresholds. The back's centering, when measured, uses the back thresholds.
- **Corners and edges**: each corner and side gets a category distribution from its pooled measurement and uncertainty. The grade rule (PSA's wording: "slight fraying at one or two corners" is 8, and so on) is enumerated exactly over the four corners or sides.
  - **Partial evidence (2.25.0):** when only a subset of corners or sides are resolved, unread features are assumed perfect (`assessed="partial"`), making the result an upper bound (`AspectGrade.is_partial`, `AspectGrade.upper`).
  - **Wear ceiling (2.25.0):** when wear observed on even a single corner or edge caps the card below its centering ceiling, a `wear_ceiling` is reported (e.g. `≤PSA 7 (partial)` or `PSA <=7 (partial)`).
- **Overall** is the weakest aspect for PSA, CGC and SGC, and the BGS rule for BGS. It is enumerated exactly. The `probabilities` are the chance of each grade given what was measured, and `confidence` is the probability of the grade shown.
- **Credible range (2.25.0):** uncertain grades are reported as an 80% credible range (e.g. `PSA 8–10`, `PSA 8-10 max`, or `PSA 8–10 (surface ?)`), replacing low-percentage modal guesses.

Until corners **and** edges are measured (at least partially with wear capping), `graded` is
False, the label reads "PSA not graded", and the centering limit is reported
only as `ceiling`. Centering alone says what the card could reach, not what it
is. In 2.24.0 such a card was labelled "PSA 10 max" and taken for a 10 (the
Probopass of 2026-10-08), which is why 2.24.1 changed the label.

With corners and edges measured but surface still missing, `complete` is False
and the label reads "PSA 9 max" (or range e.g. "PSA 8–10 (surface ?)"). That result is a ceiling: the card cannot grade
above it and may grade below it.

Certified grades (`cardcenter.learning`) still blend into the distribution as
before, and still never raise the published centering ceiling.

## The live session and Freeze

The AR session runs `analyse_view` on every measured frame (about 100 ms). Live
frames rarely resolve corners: the 2026-10-08 field log has the live view at
3-5 px/mm, with the edge strips sent but mostly not used. Two changes address
this:

- The phone now sends one full-resolution region over the whole card instead of four strips, whenever that costs about the same. The face then arrives at full resolution too, and a region with 6 mm to spare still holds the card after a small move.
- A Freeze photo taken within 6 s of the session last seeing its card is pooled into the session's evidence. It is the sharpest view the session gets.

When a new outline's first view is the other face of the last card, within 8 s,
the evidence carries across, so turning the card over grades both faces
together. Showing two different cards front-then-back inside 8 s would merge
them; Reset clears it.

The back's centering goes to its own accumulators. Before 2.24 a back seen in
the live view was pooled into the front's centering.

## What has and has not been checked

Synthetic cards with known wear are covered in `tests/test_condition.py`:

| Wear | Measured |
|---|---|
| 0.9 mm corner loss | 0.84-0.87 mm |
| 0.4 mm corner whitening | 0.33-0.39 mm |
| 40 mm of edge whitening on an 81 mm read length | 0.49 |
| 0.5 mm nick | 0.47 mm |

The synthetic tests also cover two surface cases: a scratch under two highlights
is found, and printed ink under highlights is rejected.

Real photos (owner's field log) were checked for false wear on clean cards:

- Fresh Probopass: every resolvable corner and edge reads clean.
- Motion-blurred Machamp: refused as too soft rather than graded.
- Sleeved reverse-holo Meganium: three of four corners refused where the sleeve's edge competes with the card's. Its right edge reports 0.56 mm "nicks" that can't be confirmed by eye. Sleeves and foil borders are a weak case for edges, and the multi-view median is what keeps one such view from deciding.

None of this is checked against certified grades. The category boundaries in
`condition_standards.json` (sharp under 0.15 mm, slight fraying to 0.4 mm, and
so on) are this project's reading of PSA's published words, and the file says
so. The way to correct them is certified labels, which the learning model
already accepts.

The nominal corner radius of 3.0 mm was fitted on one modern card. A design
with a different radius biases corner loss by about 0.41 times the difference.
