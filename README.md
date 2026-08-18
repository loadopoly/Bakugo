# Bakugo (cardcenter)

[![Repository](https://img.shields.io/badge/GitHub-PoodlesOfWar%2FBakugo-blue.svg)](https://github.com/PoodlesOfWar/Bakugo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Schema: cardcenter/2](https://img.shields.io/badge/schema-cardcenter%2F2-green.svg)](docs/VERSIONING_AND_CONNECTION.md)

Measures trading card centering from a photograph. Reports a ratio with a real
error bar, and a grade **band** rather than a grade.

It measures centering and nothing else. It does not assess corners, edges, or
surface, and it is not a grade prediction. Centering sets a *ceiling* on the
grade; the other three attributes decide where under that ceiling a card lands.

---

## 📚 Complete Documentation

| Document | Description |
|---|---|
| [**Architecture Guide**](docs/ARCHITECTURE.md) | Pipeline stages, optics ray-tracing, capability routing, and contamination firewall. |
| [**Getting Started**](docs/GETTING_STARTED.md) | Step-by-step setup on Android (Termux), Linux, macOS, and Windows. |
| [**CLI Guide**](docs/CLI_GUIDE.md) | Comprehensive command-line reference for single scans, batch video pans, AR, and sync. |
| [**Versioning & Connection**](docs/VERSIONING_AND_CONNECTION.md) | SemVer, schema migrations, remote sync protocols, and GitHub update checks. |
| [**API Reference**](docs/API_REFERENCE.md) | Complete Python API documentation across all modules. |
| [**Mathematical Foundations**](docs/MATHEMATICAL_FOUNDATIONS.md) | Derivations for Snell stack displacement, $\chi^2$ fusion, and Almgren-Chriss liquidation. |

---

## Why a band instead of a number

Two independent things prevent naming a single grade, and the right response to
each is different, so the tool keeps them separate and tells you which one is
binding:

| | what it is | how you fix it |
|---|---|---|
| **Measurement uncertainty** | the confidence interval on our own ratio | tripod, more light, less tilt, higher resolution |
| **Standards ambiguity** | reputable sources disagree about where the thresholds sit | you can't, from outside the grading room |

Sources genuinely disagree on PSA 10 front centering — some say 55/45, some say
60/40. PSA removed its older published standards from its site. So
`data/standards.json` carries **both** a strict and a lenient threshold for every
tier, and the reported band spans both. A card measured precisely at 57/43 comes
back as PSA 9–10 with the band attributed to *standards ambiguity*, because a
better photograph would not narrow it by a millimetre.

---

## On the web

The live app is **Bakugo** at [loadopoly.com/bakugo](https://www.loadopoly.com/bakugo/).
It runs the same `cardcenter` engine in the browser (Pyodide). Use **Live AR**
at a messy counter — glare, case glass, clutter — and still-photo Measure when
you have a clean frame. Photos stay on the device. Measurements are saved
locally and may sync metadata to the Loadopoly-OCR Supabase project when
`CARDCENTER_SUPABASE_URL` / `CARDCENTER_SUPABASE_ANON_KEY` (or the Pages
`config.json`) are set. Photos are not uploaded.

---

## Run it on an Android phone

No APK and no build. The measurement code is Python, NumPy and OpenCV; packaging
that into an app means python-for-android or Chaquopy plus a toolchain, and it
would be the same code underneath. Instead the package serves a local page and
the phone's browser is the front end. Entirely offline — useful, because shop
wifi is bad and basements have no signal.

**1. Install Termux from F-Droid** — *not* the Play Store build, which is
abandoned with frozen package repos: <https://f-droid.org/packages/com.termux/>

**2. In Termux:**

```bash
pkg install -y git
git clone https://github.com/PoodlesOfWar/Bakugo.git
cd Bakugo
bash install-termux.sh
```


**3. Start it:**

```bash
cardcenter --serve
```

Open `http://127.0.0.1:8765` in Chrome on the same phone. The capture button
opens the rear camera directly. Add `termux-wake-lock` so Android does not kill
it in your pocket, and `--lan` if you want a second device to reach it.

Runtime needs exactly **two** third-party packages, numpy and opencv; everything
else is standard library. That is deliberate — on a phone every extra dependency
is another native build that can fail. Install both with `pkg`, never pip: there
are no aarch64 OpenCV wheels on PyPI and the source build does not finish on a
phone. Tesseract is optional and only affects collector numbers.

At the counter, the two settings that matter are **holder** (pick a `case_`
variant when shooting through display glass — it is 4–6mm of n≈1.52 in the light
path) and **lens** (2× measures materially tighter than 1×).

---

## Install and use (desktop)

```bash
pip install -e .

cardcenter card.jpg --slab bgs --fov 68 --overlay out.png --json out.json
cardcenter --list-slabs
cardcenter --list-graders
cardcenter --self-test 40          # validate against synthetic ground truth
```

```python
from cardcenter import measure_centering, all_grade_bands, CaptureSpec

res = measure_centering(image, slab="bgs", capture=CaptureSpec(focal_px=2400))
print(res.worst_ratio)              # 57.8±0.76
bands = all_grade_bands(res.worst_ratio)
print(bands["BGS"].describe())
```

The overlay is the point of the visual output: it draws the **detected** frame
solid and the **ideal centered** frame dashed, on the rectified card. If the
detector locked onto the wrong edge, you can see it in one second. A number
alone is unfalsifiable by eye.

---

## At the counter (through display glass)

```bash
cardcenter case.jpg --scan --holder case_bgs --fov 68
cardcenter --video pan.mp4 --holder case_psa --stride 6 --db shop.db
```

**What this can and cannot do, measured rather than asserted.** At display-case
distance a phone resolves:

| position | lens | scale | detail per pixel |
|---|---|---|---|
| leaning on the glass | 1x | 8 px/mm | 125 µm |
| leaning on the glass | 2.5x | 18 px/mm | 55 µm |
| arm's length | 1x | 4 px/mm | 250 µm |

Edge whitening and corner wear are 50–200 µm features. At one to two pixels per
defect, **corners, edges and surface are not measurable**, and no software fixes
that. Centering survives because it is a millimetre-scale geometric measurement
across the whole card.

So live capture is a **rejection tool**. Centering caps a grade, so the tool can
rule a card *out* in seconds. It can never rule one *in*. That is still the
useful half at a counter, because centering is the single most common reason an
otherwise clean card misses its grade.

Two things make through-glass work:

- **A quality gate that refuses most frames** and says why — "glare on the glass,
  shift your angle", "too far away (3.1 px/mm), move closer". Measuring a bad
  frame is worse than measuring nothing.
- **Inconsistent frames widen the answer, they do not average into it.** Repeated
  measurements combine by inverse variance, but if they scatter by more than
  their own error bars allow, the combined uncertainty is inflated by
  `sqrt(chi2/dof)` rather than shrinking as `1/sqrt(N)`. A glare streak produces
  a *confident* outlier, not a noisy one, and averaging those is how a tool
  becomes confidently wrong.

Display glass is also 4–6 mm of n≈1.52 in the optical path — more displacement
than the slab itself (0.88 mm vs 0.27 mm at 25°). The `case_*` holders model both
layers as a stack.

---

## Batch scanning a case or binder page

One wide photo or one video pan, every card measured and ranked:

```
    card 6    50.0/50.0  CI  47.3- 52.8  PSA ceiling 10      (1 frame)
    card 8    51.9/48.1  CI  49.2- 54.6  PSA ceiling 10      (1 frame)
    card 5    57.5/42.5  CI  54.8- 60.2  PSA ceiling 8-10    (1 frame)
    ...
    card 4    83.4/16.6  CI  77.4- 89.4  PSA ceiling 3-6     (1 frame)
```

Three bugs found building this, all of which would have been silent in
production:

1. **A card's printed frame is itself a card-shaped rectangle**, so a naive
   detector reports every card twice. Fixed by rejecting nested detections.
2. **Halo detections.** A large-block adaptive threshold produces a contour tens
   of pixels *outside* the true edge, floating in blank background. It is
   *larger* than the card, so area-based selection prefers it — and because it
   adds equal margin to both sides, an 83/17 card measured as **50.8/49.2 with a
   ±0.3 error bar**. Silent, confident, and completely wrong. Fixed by requiring
   most of a quad's perimeter to sit on a real intensity gradient.
3. **Same-frame merging.** Two cards in one frame are two cards no matter how
   alike they look; identity matching by perceptual hash now only applies across
   frames.

Grouping across a video pan uses appearance, not position — centroid tracking
fails the moment the camera moves.

---

## Capture guidance

- Fill the frame; leave a small margin of plain, contrasting background on all
  four sides.
- Shoot as close to straight-on as you can. Tilt is corrected, but it costs
  effective resolution.
- Even, diffuse light. Cross-polarise for slabs and foils — glare is the single
  most common cause of a refusal.
- Pass `--fov` or `--focal-px`. Without it, camera tilt cannot be recovered, and
  slabbed measurements get an inflated error bar and a warning.

---

## Validation

`validate.py` renders cards with known border widths at random tilt, azimuth,
distance, holder type, and border-text intrusion, then checks the pipeline
against truth. Coverage is the real test: a tool reporting a 95% interval is
making a falsifiable claim.

Two independent seeds, 45 and 42 images (90 and 84 ratio measurements):

```
                       seed 7    seed 101
bias                   -0.032    +0.001   pp
RMS error               0.408     0.372   pp
95th pct |error|        0.865     0.700   pp
max |error|             1.242     1.629   pp
95% CI coverage          98.9      98.8   %     (want >= 95)
z-score std              0.81      0.76         (want ~1)
```

Unbiased, sub-half-percentage-point RMS, and intervals that slightly
over-cover — which is the safe direction to err.

**Multi-card scanning**, 32 cards across 4 randomised scenes at 9 px/mm with
rotation, random border and art colours:

```
bias                   -0.040 pp
RMS error               0.908 pp
95% CI coverage          100 %
refused                  6/31
```

Coverage there was initially **80%** — the error bars were lying. The missing
term is that a boundary located in a wide, lower-resolution frame carries about
a pixel of *placement* bias, which a line-fit residual does not capture: the
residual measures how straight an edge is, not whether it sits a pixel inside or
outside the true edge. Adding a 1 px systematic floor restores full coverage at a
z-score spread of 0.70. Raising it further only wastes precision.

**What this does not validate.** The synthetic renderer shares its refraction
code with the solver, so these numbers confirm the *geometric chain* — homography
fitting, pose decomposition, the apparent-corner iteration, coordinate mapping —
not the physics. The physics is validated separately and genuinely independently
in `tests/test_optics.py` by numerically minimising optical path length
(Fermat's principle) and comparing. And none of it validates *real cards*: print
variation, holo foil, glare, rounded die-cut corners, and borders that shade
continuously into artwork are all absent. **Synthetic accuracy is an upper bound
on real accuracy, never an estimate of it.**

---

## Three findings that changed the design

### 1. Refraction barely matters for the ratio

The intuition that a slab's acrylic displaces the apparent card position is
correct, and the displacement on a *single border width* is real — 0.27mm
through 1.6mm of PMMA at 25°. But centering is a **ratio**, and the inner frame
and the card's outer edge sit only ~3mm apart, so they are displaced by nearly
the same amount. It is common-mode and it cancels. Only the *gradient* of the
displacement across those 3mm survives.

Measured effect on the ratio, BGS slab at 25° tilt:

| working distance | angular spread across card | effect of correction |
|---|---|---|
| 90 mm | 35° | 0.34 pp |
| 130 mm | 26° | 0.21 pp |
| 220 mm | 16° | 0.08 pp |
| 400 mm | 9° | 0.02 pp |
| 900 mm | 4° | 0.002 pp |

At normal phone working distance it is below the detection noise floor. It
scales with how **close** the camera is, not with tilt alone. The correction is
kept because it is cheap and it does matter up close — but it is not a moat, and
anything built on the premise that it is one is built on sand.

### 2. The commonly cited displacement formula is the wrong one

The textbook lateral displacement through a parallel plate,
`d = t·sin(θ₁−θ₂)/cos(θ₂)`, measures the perpendicular offset between the
incoming and outgoing ray *lines*. What card measurement needs is the in-plane
displacement on the card surface:

```
δ = t · (tan θ₁ − tan θ₂)  = d / cos θ₁
```

Using the perpendicular form under-reports by 9% at 25° and 23% at 40°.
`test_perpendicular_formula_relationship` pins the distinction so nobody
"corrects" it back.

### 3. Slab thickness is routinely double-counted

Air gaps and inner sleeves contribute **zero** lateral displacement — with n=1
the two tangent terms cancel exactly. Only the solid plastic face wall counts,
around 1.6mm for BGS-style acrylic, not the ~3.25mm total slab depth. Using
total depth overcorrects by roughly 2×.

`SlabSpec.acrylic_thickness_mm` is documented as the face wall, ships with
generous uncertainties because these are estimates rather than manufacturer
specs, and `--list-slabs` tells you to measure your own holders with calipers.

---

## How it works

```
image
  -> outer card quad          subpixel, by fitting lines to each edge
  -> camera pose              homography decomposition (needs focal length)
  -> refraction solve         iterated, because the corners are displaced too
  -> rectify                  fronto-parallel at a known px/mm
  -> inner frame per side     per-column ΔE transitions in CIELAB, median-combined
  -> map back to true coords
  -> ratios + error budget
```

Two choices carry most of the robustness:

**Line fits, not contour vertices.** Contour vertices are pixel-quantised and
land wherever the polygon approximation broke. Fitting a line to all the points
along each side and intersecting adjacent lines gives subpixel corners *and* a
residual, which becomes an honest uncertainty — a card edge really is straight,
so any residual is noise or a bent card, and both are worth telling the user.

**Median across columns, not mean.** Copyright lines, set symbols, and holo
patches intrude into borders constantly, but rarely across more than half a
side. A median survives them; a mean does not. The *spread* of the per-column
transitions is not discarded — it becomes the uncertainty on that border, and
its slope becomes a print-rotation warning.

**Scale invariance.** The ratio `L/(L+R)` is invariant to overall scale, so an
error in the assumed physical card size cancels completely and never enters the
error budget. Errors locating either individual edge do not cancel, and those
dominate.

---

## Refusals

The tool raises `DetectionError` rather than returning a low-confidence number.
It refuses on full-bleed cards (nothing to measure), pure noise, no card-shaped
quadrilateral, borders indistinguishable from artwork, and detection confidence
below threshold. Roughly half the test suite asserts these refusals.

The failure mode that matters for a measurement tool is not a loud wrong answer.
It is a quiet plausible one.

---

## Tests

```bash
python -m pytest tests/ -q       # 321 tests
python validate.py 60            # Monte Carlo coverage check
```

`tests/test_optics.py` is the one worth reading. Everything else is checked
against a renderer that shares code with the solver, so it can only catch
implementation bugs. The Fermat ray trace derives the displacement a completely
different way and would catch a physics error.

---

## Known limitations

- **Single face per run.** Front and back are measured separately; pass
  `--face back` for the more lenient back tolerances.
- **Standard card size assumed** (63.5 × 88.9 mm). Tobacco-size vintage,
  oversized, and die-cut cards are not handled.
- **One rectangular printed frame assumed.** Cards with concentric frames,
  gradient borders, or artwork bleeding into the border will refuse or mislead.
- **Slab thickness defaults are estimates**, not manufacturer specs.
- **CGC and SGC tables are weakly sourced** and flagged `confidence: low` in the
  data file. They are inherited/interpolated, not published. Verify before use.
- **Never validated on real photographs.** Everything above is synthetic.

---

## Offers, and why expected value is the wrong headline

A centering ceiling is an upper bound, not a grade. Expected value is therefore
**not** `price(ceiling)` — it is `sum P(grade) * price(grade)` over grades at or
below the ceiling, and `P(grade)` has to come from somewhere real. There are
three honest sources: the population report for that card, your own submission
history, or an assumption you will defend out loud. `GradePrior.uninformative()`
exists, is uniform, is almost certainly wrong, and stamps a warning on every
number it touches.

The offer uses the **pessimistic** end of the band. Offering as though the
optimistic ceiling were certain means paying full price for our own uncertainty.

The headline outputs are **loss probability** and the **10th-percentile
outcome**, with expected value alongside rather than in front. At a counter you
are making a bet with a fat left tail — the card lands two grades under the
ceiling and grade spreads are routinely 5–20x. A positive expected value with a
40% loss probability is a bad trade to make forty times in an afternoon.

There is **no built-in price data and none is fabricated.** Configure a real
source or the valuation refuses. The eBay class points at the genuine Browse
endpoint (`api.ebay.com/buy/browse/v1`), requires a real OAuth token, and refuses
to substitute *asking* prices for *sold* prices — sold comparables need the
Marketplace Insights API, which is separately approved. That substitution
silently inflates every offer. `CsvPriceSource` rejects rows without a sold date
for the same reason.

---

## The training loop, and why it is left open

The tempting design is: the model estimates a grade → the marketplace shows that
estimate → participants confirm it → confirmations become training labels → the
retrained model sets prices. Every arrow is reasonable alone. The whole is
degenerate, because **there is no external anchor anywhere in it.** A model
trained on opinions it originally seeded converges on internal consistency, not
on truth, and it drifts toward whatever is most profitable for the people
voting. When those numbers set prices, the drift *is* the product.

The only label that breaks the loop is a grade issued by a grading company on
that physical card, identified by cert number. Slow and expensive, which is
exactly why the shortcut is tempting.

So the store records provenance that cannot be washed off:

| kind | what it is | trains? |
|---|---|---|
| `CERTIFIED` | grader-issued, cert number required | yes |
| `SELF_REPORTED` | owner says it graded X | evaluation only |
| `MARKETPLACE_VOTE` | crowd opinion / confirmatory poll | no — this is sentiment |
| `MODEL_PREDICTED` | our own output | no — this is nothing |

`export_training_set()` returns certified rows only. Anything else raises unless
you pass `acknowledge_contamination=True`, which is then stamped into the
manifest so a model trained on crowd votes can never later be described as
having been trained on grades. `circularity_report()` states plainly what
fraction of the pool is independent.

Those certified rows are what expand grade prediction. `GradeOutcomeModel`
is a Dirichlet-multinomial over issued grades, stratified by grader and
centering-ratio band. Import / `--sync-url` / `--ingest-grades` rebuild it
from the certified export; with zero observations `predict_overall_grade`
is identical to the published-table heuristic. The model's own predictions
are never written back as labels.

Marketplace confirmations are still worth collecting — as sentiment, as a
prioritisation signal for which cards to actually send in, and as evaluation
data. They are not grades.

---

## Identification: the card is easy, the printing is not

Catalogs exist and they are good — `ScryfallCatalog` queries the live API, no
key needed, and returns every printing with collector numbers, finishes and
market prices. Getting card data was never the hard part.

**The hard part is which printing.** Scryfall lists 131 printings of Sol Ring,
spanning **$1.20 to $3,521 — a 2,934x range.** Naming the card is easy: art and
title are large, high-contrast, and match robustly under ORB features even
through glass at an angle. Naming the *printing* determines the price, and
printings differ by features that are millimetres tall:

| feature | printed size | 4 px/mm | 8 px/mm | 18 px/mm |
|---|---|---|---|---|
| collector number | 1.5 mm | 6 px ✗ | 12 px ~ | 27 px ✓ |
| set symbol | 3.5 mm | 14 px ✓ | 28 px ✓ | 63 px ✓ |
| border / full-art | whole card | ✓ | ✓ | ✓ |
| foil vs non-foil | *not spatial* | ✗ | ✗ | ✗ |

So identification returns an `Identification` that is explicitly **RESOLVED** or
**AMBIGUOUS**. When ambiguous it reports the full candidate price spread and
refuses to collapse it:

```
identified: Sol Ring  (140 verified features)
  PRINTING AMBIGUOUS: 131 candidate printings
  raw market spans $1.20 - $3,521.30  (2934x)
  cannot read at 8.0 px/mm: collector_number, finish
  WARNING: candidate printings differ by 2934x in price.
           Do not make an offer until the printing is pinned down.
```

A confident wrong printing is worse than a visible question, because it produces
a specific wrong number instead of a prompt. The realistic path at 1x is that a
human reads the collector number and types it; supplying it collapses the
ambiguity to one $1.79 card. A number that matches **nothing** raises rather
than falling through — silently ignoring it and resolving to whatever remains
returns a confident identification of the wrong card, and the operator believes
they disambiguated it.

Foil gets its own permanent warning: through display glass, foil and glare look
alike, and foil usually carries the higher price. Finish is never inferred from
a through-glass frame at any resolution.

Matching uses ORB with RANSAC geometric verification rather than perceptual
hashing. A hash compares whole images and fails on exposure change, white
balance, residual rotation and glare — all four of which a display case
guarantees. Local features survive partial occlusion, which is the glare case
exactly. Geometric verification matters because cards from one set share frames,
symbols and typography, and will produce dozens of spurious raw matches.

---

## OCR: gated, and constrained by the catalog

Collector-number OCR runs on local Tesseract behind two constraints. The
resolution gate is set from **measurement, not assumption** — rendering numbers
at a range of glyph heights with through-glass blur and noise, reading them, and
snapping to the catalog:

| glyph px | 8 candidates | 30 candidates | 131 candidates |
|---|---|---|---|
| 6 px | 38% | 42% | 17% |
| 9 px | 75% | 83% | 75% |
| 12 px | 100% | 100% | 92% |
| 18 px | 62%* | 92% | 100% |

<sub>* small-sample artefact: with 8 candidates they are all single digits, which collide heavily under edit distance.</sub>

What sets the floor is not the accuracy but the **error mode**. Below 12 px a
failed read does not come back empty — it snaps to a real but *wrong* collector
number. At 9 px roughly one reading in five is confidently wrong. So the gate
sits at 12 px, with a marginal band to 20 px that flags for verification.

This also revised the earlier claim upward: leaning on the glass at 1× gives
12 px, which measurement puts at the usable floor rather than below it.

**The catalog constraint is what makes OCR safe here.** This is not open-vocabulary
reading — the answer must be one of the N collector numbers this card actually
has. That constraint is applied twice: the engine's charset is restricted to
characters appearing in real candidates, and the reading is then snapped to a
candidate only if it is closer to exactly one of them than to any other. A read
of `2S3` resolves to `263`; a read of `11` against `{1, 111}` stays ambiguous; a
read matching nothing within edit distance 1 is reported unread rather than
forced. In testing, Tesseract misread a `5` as `3` and the constraint recovered
the correct printing anyway.

Note the direction of the argument: a *better* OCR engine does not help below the
gate, because the information is not in the pixels. A vision LLM asked to read
6 px digits returns a plausible number rather than a refusal, which is strictly
worse than reporting ambiguity. Here the dumber local engine is the safer one,
because it fails visibly.

---

## AR mode, and calibrating against a caliper

### The loop is shaped by measured throughput

| long side | detect | full measure | accuracy |
|---|---|---|---|
| 2400 px | 1771 ms | 2086 ms | 0.20 pp |
| 1200 px | 300 ms | 364 ms | 0.06 pp |
| 540 px | 65 ms | 119 ms | 0.23 pp |

Detection dominates, and accuracy is flat from 1200 px up — so measuring at full
sensor resolution buys nothing and costs 6×. The loop splits accordingly:

- **Track** every frame by searching along the previous quad's edge normals.
  Cropping and re-detecting only bought 1.4×, because the card fills most of the
  frame and the cost is the multi-strategy contour search, not the pixel count.
  Normal search is **10× faster — 5.9 ms, 168 Hz** — and agrees with full
  detection to 1.4 px. It refuses rather than drifting, because a tracker that
  has wandered onto something else keeps the session averaging.
- **Measure** at 1200 px, only on gated frames, a few Hz. Results accumulate in
  the existing inverse-variance combiner, so the band tightens while you hold
  still. A simulated hand-held sequence settles at 56.6/43.4 against truth 56.67.

One bug worth naming: the quality gate was initially checking resolution at
*track* scale, where a card is ~5 px/mm — below the usable floor. It rejected
every frame while the 1200 px measurement would have been fine. The gate now
judges the resolution the measurement will actually have.

### Caliper calibration

Hold a caliper open to a known reading in frame, tap the jaw tips (or let it find
them), and the session's scale is fixed to the caliper's own precision. It is
updatable and it ages: VIO scale drifts, so a calibration's uncertainty widens
with time and goes stale after two hours.

**Two things will ruin it, and both are quantified rather than hand-waved.**

*Coplanarity.* Scale is apparent size over true size, and apparent size goes as
1/distance. A caliper held 10% nearer than the card makes every dimension 10%
wrong — **6.3 mm on a card width**, a hundred times the thing being measured.
Resting the caliper on the same surface is the whole measurement, not a nicety.

*Pixel baseline.* A caliper reads to ±0.02 mm, but that precision is thrown away
if the jaws cannot be *located* to better than a pixel or two. At a 500 px
baseline the localisation term is 0.42% — **worse than simply using a bank card**
(0.152%). The caliper only wins when its gap spans enough pixels, so calibrate at
full sensor resolution with the gap filling the frame. The tool says so when it
is in the losing regime.

Automatic jaw detection had its own trap: a clean caliper shot produces **four**
gradient peaks, not two, because each jaw has an outer and an inner face. Taking
the two strongest can straddle a single jaw and silently halve the scale. The
detector instead picks the pair bracketing the widest flat region — the inner
faces, which is what a caliper actually closes on.

`verify_calibration_against_card` then measures the card itself: a standard card
is 63.5 mm, so a fresh calibration that disagrees by more than a couple of
millimetres is wrong. It cannot distinguish a bad calibration from a genuinely
trimmed card, and it says so rather than picking one.

---

## Edge shadow: the worst bug in this project, found by a question about shadows

A card is not a plane. It is about 0.30 mm thick, and lit obliquely it casts a
shadow of its own edge onto whatever it sits on — **on the side away from the
light only.**

Nearly every other error here is common-mode and cancels in a ratio. This one
adds to one border and not its opposite. Measured in simulation on a *perfectly
centred* card:

| light elevation | shadow cast | measured ratio | error |
|---|---|---|---|
| 60° | 0.17 mm | 50.8 / 49.2 | +0.8 pp |
| 45° | 0.30 mm | 50.8 / 49.2 | +0.8 pp |
| **30°** | 0.52 mm | **82.8 / 17.2** | **+32.8 pp** |
| 20° | 0.82 mm | 76.7 / 23.3 | +26.7 pp |

At 30° a perfectly centred card reads 83/17, and the reported error bar misses
truth by **seven sigma**. Eighty times the package's RMS error, caused entirely
by where the lamp is, and invisible until someone asked what shadows actually do.

**Detecting it took two attempts, and the first failure is the instructive one.**
The obvious test — compare background brightness outside opposite edges — fails
exactly when it matters, because by then the detector has already swallowed the
shadow: the boundary it found is the shadow's *outer* edge, so sampling outside
finds ordinary background on all four sides.

The shadow is *inside* the detected quad, and it is darker than both the card and
the background, being background with the light taken away. Measured across one
edge at 30°: card border 188, backdrop 34, **shadow 14**. That third level on one
side only is unambiguous.

### Refusing was also wrong

The first fix declined to measure below 55° of light elevation. That is useless
where the tool is meant to be used: shop lighting is whatever the case has,
usually LED strips at a low angle behind glass, and you cannot move it. A tool
that refuses below 55° refuses nearly every real card.

The shadow is *measurable*, so it gets **subtracted** instead. The true card edge
is the band's inner boundary, not its outer one:

| light elevation | before any fix | refuse-only | detect and correct |
|---|---|---|---|
| 45° | +0.8 pp | measured | +0.8 pp |
| 30° | **+32.8 pp** | refused | **+0.7 pp** (σ 1.4) |
| 20° | +26.7 pp | refused | **+0.1 pp** (σ 2.5) |

The correction carries its own error term — a penumbra is not a step, so the
band's inner boundary is soft — which is why σ widens from 0.42 to 1.4–2.5. The
interval still covers truth. The tool only gives up when the band is wide enough
to overlap the printed border, and then it suggests moving around the case rather
than moving the shop's lights.

### And the display-case case was never the bad one

The 0.30 mm figure is a raw card lying loose on a table. A card held against a
slab's inner well has roughly half that gap:

| situation | gap | shadow at 30° |
|---|---|---|
| raw card loose on a table | 0.30 mm | 0.52 mm |
| slabbed, held against inner well | 0.15 mm | 0.26 mm |
| card in a penny sleeve | 0.05 mm | 0.09 mm |

A slabbed card in a case measures identically at 90°, 45°, 30° and 20° — the
shadow never gets large enough to trip anything. The severe case is a loose raw
card on a table, which is the one situation where you *can* move the light.

Parallax was the other candidate: a shadow lies on the backing plane and the card
edge on the card plane, so AR camera motion should separate them by depth.
Measured, it buys only 0.3–2.3 px of differential shift across a realistic
viewpoint change — comparable to the edge-localisation noise, so too weak to rely
on. Subtracting the measured band is the stronger method and it works in a single
frame.

---

## Absolute dimensioning, and the quantum floor

Two additions, and a clear statement of what does not transfer.

### Photogrammetry: QUIPU's mechanism is real, its scale source is not available here

`geospatial_relation.unproject_pixel` is a standard pinhole ray-cast intersected
with a ground plane. Its metric scale comes from a **known sensor altitude**, and
its error model is `sigma = max(0.3, 0.05 x range)` — a 30 cm floor. Sound for a
drone at tens of metres; a trading card is 63.5 mm wide, and a phone over a
display case knows neither its altitude nor its distance.

The underlying fact is not an engineering gap: **a single uncalibrated photograph
is scale-ambiguous.** Metric size needs either a known camera distance or a known
length in the scene. So the module supplies a coplanar reference object, and
reports the resulting budget honestly:

| reference | tolerance | scale error | on 63.5 mm |
|---|---|---|---|
| ISO ID-1 bank card | ±0.13 mm | 0.152% | ±0.097 mm |
| printed target, unverified | ±0.30 mm | 0.35% | ±0.22 mm |
| same, caliper-measured | ±0.02 mm | 0.023% | ±0.015 mm |

**What this buys is trim detection.** Centering is a ratio, so scale cancels and
none of this affects it. But a *trimmed* card — cut down to improve centering —
is undersize, every grader rejects it, and it is invisible to a ratio because
trimming **improves** the ratio. Only absolute measurement sees it.

A finding from building it: past about 0.05% scale error the reference stops
mattering and **image resolution takes over.** At 10 px/mm a half-pixel edge
uncertainty is already 0.071 mm, which is larger than a caliper-verified target
contributes. `Dimensions.limited_by` reports which term binds, because buying a
better ruler at that point buys nothing.

### The sunlight entanglement paper: real work, no path to this problem

Li et al., *Optica* (2026) concentrate sunlight through a Fresnel lens and glass
cone into a ppKTP crystal in a Sagnac interferometer, and get polarization-
entangled pairs from SPDC — concurrence 0.905 ± 0.053, Bell S = 2.54 ± 0.22. The
motivation is energy efficiency: replacing laser pumps.

It does not reach card measurement, for mechanical reasons rather than
squeamishness:

1. The entanglement is **created in the crystal**. Sunlight is the pump and is
   annihilated in down-conversion. Light reflecting off cardboard has undergone
   no nonlinear process, so there is no entanglement present to lose.
2. Reflected imaging light is thermal. It shows photon bunching (Hanbury
   Brown–Twiss), which is a classical intensity correlation carrying no extra
   positional information.
3. A CMOS sensor is an **intensity** detector; it discards phase and
   polarization. Demonstrating entanglement needs coincidence detection across
   two separated detectors with independent basis choice.
4. "Interstitial pixel space" is fill-factor dead area and microlens. Photons
   there are absorbed or redirected — ordinary optical loss, already modelled as
   quantum efficiency.

### What *is* quantum here, and why it settles the question

Photon shot noise. Arrivals are Poisson because light is quantised, and that sets
the Standard Quantum Limit for intensity imaging. The Gaussian bound was an
approximation to it; `shot_noise_fisher_edge` computes the real thing with
`I = Σ (∂μ/∂θ)² / μ` in photoelectrons.

Running the variance budget on a normal capture:

```
total reported uncertainty : ±0.4300 pp
  photon shot-noise part   : ±0.003478 pp  (0.0065% of the variance)
  everything else          : ±0.4300 pp

sensor noise vs shot-noise floor: 1.2x
  shot-noise limited: photons are the binding noise source

If photon noise were driven to ZERO — a perfect detector, or any
quantum-enhanced scheme reaching Heisenberg scaling — the reported
uncertainty would improve by 0.00327%.
```

The sensor is *already* shot-noise limited, and photon statistics contribute
0.0065% of the variance. A quantum-enhanced card scanner would improve the answer
by three parts in a hundred thousand. That is arithmetic, not an opinion, and it
is why the effort belongs elsewhere.

The same machinery falsifies error bars in the other direction:
`shot_noise_consistency` flags measured noise *below* the Poisson floor, which is
impossible and means the image was denoised — and denoising invents detail.

### QUIPU's compaction claim

The most recent commit enforces "20-byte binary tensor shard packing for
1,360,000x compaction factor". Reading it: `pack_weyl` writes 5 × Float32 LE into
20 bytes, and `unpack_weyl` returns those five floats. That is a **fixed-size
summary**, not compression — the tensor is not recoverable, so the ratio is
(input size)/(20 bytes) for any input, and it would be equally "1,360,000x" on
noise. The distinction matters directly here: you cannot decompress pixel loss
back into truth, because the information was never stored. That is exactly what
the Fisher bound above quantifies.

---

## Appendix: two-view filament reconstruction (negative result)

`channel.py` implements two-observer 3D reconstruction of a self-luminous
filament from photometry alone. **It does not work at realistic ranges.** It
ships with tests because the geometry is exact and the failure is not obvious,
and someone — including a later me — would otherwise rebuild it.

**The physics is right.** Brightness gives chord length; chord length gives the
angle between the local tangent and each line of sight. Two observers put the
tangent on two cones, which intersect in ≤2 directions. Verified: brightness
inversion exact to 1e-9, and the true tangent is in the solution set at every
point to within 0.01°. Two observers *are* sufficient in principle.

**Three things kill it, and they are independent.**

**1. The sign is unmeasurable.** Chord length depends on |cos θ|. A segment
tilted +30° toward you and one tilted −30° away are photometrically identical —
there is a test asserting this. One bit of physics (lightning runs cloud-to-
ground) fixes the seed. Complexifying the angle does not: θ=√−1 gives
cos θ ≈ 1.543, so |cos θ| > 1 and no unit vector satisfies the constraint.
Continuation adds a continuous dimension where a discrete bit is missing.

**2. Branch tracking fails mid-curve, even with perfect data.** Across 12 scene
phases, noiseless recovery is *bimodal*: 7/12 at 0.000°, 5/12 above 20°, worst
63.6°. The one-bit prior fixes the start; nothing fixes a wrong branch taken
halfway along, which then propagates perfectly. An earlier single-seed check
reported 0.001° and was taken as proof it worked — it was one of the 7.

**3. Conditioning beats any achievable SNR.** The cone solve carries a
`1/(1−d²)` factor with `d` the cosine between the two lines of sight. At 2 km
range it is ~213 at 140 m baseline. Measured across baselines 140–1800 m and
SNR gains 1×–16×:

| baseline | ×1 SNR | ×4 SNR | ×16 SNR |
|---|---|---|---|
| 140 m | 26% / 42° | 24% / 36° | 37% / 45° |
| 900 m | 38% / 42° | 53% / 43° | 79% / 44° |
| 1800 m | 37% / 41° | 59% / 43° | 87% / 47° |

*(resolved fraction / median tangent error)*

Better photometry buys **coverage and no accuracy**. A 16× SNR gain is ~256
frames averaged, and tangent error stays at 42–47° throughout. Sub-pixel
interpolation across neighbours is real and does help — it is why resolved
fraction climbs — but it cannot outrun the amplification.

**What was kept from the DCQE pruning proposal.** The STP-Torus tube gap
(QUIPU v0.25.0's `1 − cos(h_t−h_r, h_r−h_s)`, which is discrete curvature and
transfers directly), the bilateral evidence rule (already implemented as
inverse-variance fusion in `geospatial_relation.relate_detection`), and the
Poisson–Gaussian noise model. Dropped: the delayed-choice eraser — DCQE shows no
interference in the unsorted marginal, so it is post-selection on data already
held, not retrocausation, and supplies no mechanism; CAT/W-state entanglement,
since these are classical hypotheses; and the G2-torsion bound, which has no
referent here.

**Partial recovery — and a corrected claim about why.** Greedy branch tracking
solves 7/12 scenes exactly. A pool of selectors arbitrated by photometric refit
solves **10/12**. An oracle over the pool is 12/12, so the correct assignment is
always present and the entire problem is selection.

I initially attributed this to graduated non-convexity (Yang et al., RA-L 2020)
and that was wrong. Diagnosing why the annealing rate changed nothing: every
candidate set contains both **t** and **−t**, and the smoothness term uses
`abs(dot)` because sign is meaningless for curvature — so a ± pair scores
identically at every μ and the continuation has nothing to act on. Rates 0.50,
0.70 and 0.85 all return the same answer. There is a test pinning that.

The real finding is the **selection criterion**. Choosing by smoothness recovers
7/12; choosing by photometric refit against *both* views recovers 10/12. That is
the difference between a proxy and the data. The honest label for the mechanism
is multi-start with data-driven arbitration, not GNC. It leaves the conditioning
result untouched — accuracy under noise is unchanged.

**What would actually work**, in order of how much it buys: a third observer
(three cones intersect uniquely, killing the branch problem outright);
inter-channel occlusion at branch crossings, which gives signed depth ordering
from a single frame; and nanosecond-gated hardware, which is a different
instrument class rather than better processing.

---

## Learning from what it sees — and the QUIPU review

I read QUIPU's source (19k lines across `src/quipu`). Three things are worth
taking; the central one is not.

**Worth taking.** `radam_optimizer.radam_step` documents itself as *"a strict
mathematical superset… when all extension knobs are at their identity values,
produces bit-for-bit the same trajectory as vanilla Adam. Verified by
test_radam_optimizer.py."* That discipline is adopted here as a hard contract:
with zero observations the posterior decoder must reproduce the edit-distance
snap exactly. **Measured: 400/400 untrained decodes against a 131-printing
catalog return a minimum-edit-distance candidate.** Also taken: the delta-space
scaling in `neural_plasticity._smooth_dial`, and the `local_store`/`brain_kv`
persistence pattern.

**Not worth taking: rADAM itself.** It is sound for its problem — a
high-dimensional non-convex surface with no closed form, where momentum, noise
injection and rectification earn their keep. That is not this problem. The three
things this system must learn are:

| estimand | model | posterior |
|---|---|---|
| character confusion | Dirichlet-multinomial | exact conjugate |
| which printings appear | Dirichlet | exact conjugate |
| Almgren-Chriss η, γ | Normal-inverse-gamma | exact conjugate |

All three have closed forms. A stochastic optimizer would converge slower, add
noise to an exact answer, and — decisively — discard the posterior **variance**,
which is the most important output here. A schedule computed from an η estimated
to ±300% is not a schedule, and only the Bayesian treatment tells you that. The
heartbeat-momentum and toroidal-phase extensions have no defined meaning for
these estimands.

### What learning can and cannot do about pixel loss

**It cannot recover lost pixels.** Information not captured is not in the file,
and no amount of training changes the channel capacity of a 6px glyph. The
resolution gate stays exactly where measurement put it — there is a test
asserting that 500 training observations do not reopen it.

What learning does is improve **decisions under** pixel loss:

- A learned confusion model replaces edit distance with a real likelihood. Edit
  distance treats `8→0` and `8→4` as equally likely; they are not.
- A learned encounter prior replaces "uniform over 131 printings" with what
  actually walks through the door.

Both directions are tested, including the one that matters most: **learning must
be able to raise doubt, not only confirm.** A reading that edit distance resolves
uniquely becomes uncertain when the prior disagrees.

### One real bug this surfaced

The first likelihood summed per-character log-probabilities. Every character
contributes a negative term, so a **shorter candidate scored higher purely by
having fewer terms** — a reading of `11` preferred printing `1` over `111` for
reasons unconnected to the image. The fix is to score each candidate as a
likelihood *ratio* against the read-perfectly null, so a correct character costs
zero regardless of how many there are and lengths become comparable. That also
made the identity reduction exact rather than approximate.

### The circularity rule, one layer down

Only **verified** observations update these models. A confusion model trained on
its own output converges to certainty about its own biases — the same degenerate
loop as marketplace confirmation, and more insidious because it is invisible.
There is no code path that feeds a decode back in.

---

## Inventory liquidation (Almgren-Chriss)

For **inventory**, not single cards. Eleven copies from a case break is a real
execution problem: dumping them all this week depresses the price on all eleven,
holding six months exposes you to six months of market. That is exactly the
tradeoff Almgren-Chriss solves, and the trajectory

```
x(t) = X · sinh(κ(T − t)) / sinh(κT)
```

with half-life `1/κ` is the useful summary. The efficient frontier — cost versus
risk across urgency levels — is the actual output; there is no single right
schedule, only a curve and a preference.

Three limits are enforced rather than documented:

- **One card raises.** There is no schedule to optimise; use the liquidity
  estimate instead.
- **Impact parameters are required, not defaulted.** `η` and `γ` must be
  calibrated from your own fills — `ImpactEstimator` does this online by exact
  Bayesian regression of realised slippage on cumulative volume and selling
  rate, recovering known parameters to within 15% and reporting posterior
  uncertainty. Updates are order-independent, non-physical fits (negative η)
  fall back rather than generating a nonsense schedule, and a relative σ above
  50% marks the result uncalibrated. `ImpactParameters.assumed()` exists so the
  machinery can be exercised, keeps `calibrated=False`, and stamps every result
  with *"the SHAPE of this schedule is meaningful; the dollar figures are not."*
- **`check_feasibility` tests the schedule against observed volume.** This is the
  check that keeps the model honest for collectibles. Almgren-Chriss assumes a
  continuously traded asset with depth; cards arrive as a Poisson process. If the
  schedule peaks at 2 sales/day and the market has absorbed 0.02/day, the
  trajectory is arithmetic rather than a plan, and it says so.

---

## Flip period from trading volume

I was wrong to file this under "not a software problem". Sold-comp timestamps
*are* volumetric data and time-to-sale follows from them directly. The
distinction that matters:

- **Computable** — a distribution over days-to-sale at a given asking price,
  with an honest interval, and the holding discount that follows
- **Not computable** — "you can flip this immediately for $X". That is a
  market-making function requiring capital and inventory risk, not a
  calculation.

Sales arrive as a Poisson process. Observing *n* sales over window *T* gives the
Gamma(n + ½, T) posterior, so the marginal wait is Lomax:

```
P(sold within t) = 1 - (1 + t/T)^-(n + 1/2)
```

That form matters. With two sales, the naive `T/n` point estimate is wildly
overconfident; the Lomax tail is heavy and honest about it. Below five sales the
module **refuses to report a flip period at all** and calls the card illiquid:

```
=== THIN: 3 sales over 2 years ===
  THIN MARKET: 3 recorded sale(s). No flip period is reported
               because none would mean anything.
  P(sold within 30 days): 13%
  P(sold within 90 days): 33%
  observed volume       : 0.1 sales/month
```

Price elasticity is the weak link and is labelled as such. It is fitted from the
comps when there are enough sales across enough price spread, and flagged
`ASSUMED, not measured` when not. Two other biases are surfaced rather than
buried: inferring the observation window from first-to-last sale understates it
and therefore **overstates liquidity**, and rows without a sold date are dropped
because an asking price is not a sale.

---

## What is still not built

- **A nominal price for the Bakugo card.** No comps, no number. Supply sold
  comparables and the valuation runs; without them it refuses, which is the
  entire point.
- **Set-symbol classification.** Collector-number OCR is implemented (below);
  identifying the set by its symbol glyph is not.
- **Corner, edge, and surface grading.** Not at these resolutions. See the
  table above.
- **A guaranteed instant buyer.** See the liquidity section.

---

## A note on what this deliberately does not do

It does not emit a single numeric grade, and nothing in it is designed to be
piped into a pricing engine, a minting contract, or any other system that turns
an estimate into money. A grade band with an explicit error bar and a named
dominant uncertainty is both more honest and more useful than a hard number —
and a hard number attached to real financial output is a liability wearing a
measurement's clothes.

MIT licensed.
