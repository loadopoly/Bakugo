# Bakugo capability-expansion research plan

## Objective

Expand what Bakugo can reliably observe, while preserving honest uncertainty,
abstention, provenance, and practical mobile operation. Larger AI models alone
are not a substitute for measurable improvements.

Bakugo already supports centering measurement, live AR capture, card
identification, and market analytics. This plan proposes research, not claims
that the proposed capabilities have already been validated.

## Priority research areas

| Priority | Research | Capability unlocked | Evidence of success |
|---|---|---|---|
| 1 | Real-world measurement calibration: compare results with independently measured cards across phones, sleeves, slabs, glare, and tilt. | More trustworthy measurements in shops and through glass. | Lower measurement error; reported 95% intervals achieve approximately 95% coverage, including capture-condition slices. |
| 2 | Exact-printing recognition: investigate compact visual embeddings combined with OCR, collector numbers, set symbols, and foil cues. | Distinguish reprints, editions, and variants, not just species names. | Higher exact-printing accuracy at a fixed false-accept rate; explicitly evaluate unseen printings and abstention. |
| 3 | Active capture guidance: study which next action, such as changing angle, switching lens, refocusing, or capturing another frame, reduces uncertainty. | AR that guides users toward usable measurements rather than repeatedly refusing. | Shorter time to a trustworthy result without increased false acceptance. |
| 4 | Close-up condition inspection: research macro imaging, controlled directional lighting, and multi-view capture for corners, whitening, scratches, and dents. | Condition evidence beyond centering. | Defect detection validated against expert annotations, with minimum resolvable defect sizes stated. |
| 5 | Reference-based alteration screening: validate physical dimension measurements against manufacturer and production-era variation. | Better screening for undersized or potentially trimmed cards. | Detect known alterations while minimizing false flags on legitimate manufacturing variation. |
| 6 | Outcome and market calibration: study verified submission outcomes and licensed sold-price data, accounting for selection bias, fees, and sparse trades. | More defensible submission decisions and downside-risk estimates. | Calibrated grade probabilities and valuation errors on held-out outcomes. |

## Shared research foundation

- Build a consented benchmark with independent ground truth.
- Separate training and evaluation data by physical card, device, and capture
  session to prevent leakage.
- Include difficult captures and unfamiliar cards, rather than evaluating only
  clean images or previously confirmed identities.
- Preserve abstention and provenance: user guesses and model predictions must
  not become verified training labels.
- Audit the documented QUIPU feedback loop before expanding learned priors.
  The learning documentation flags possible circularity from the system's own
  collector-number readings.
- Measure latency, battery consumption, bandwidth, and offline operation
  alongside accuracy.
- Compare proposed methods against the current pipeline on held-out data before
  promoting shared artifacts.

## Recommended sequence

1. Establish the benchmark and validate measurement uncertainty across capture
   conditions.
2. Evaluate exact-printing recognition against the current OCR and embedding
   confirmation paths.
3. Use capture experiments to prioritize guidance actions that demonstrably
   improve measurement quality.
4. Pursue condition inspection as a separate controlled-capture mode.
5. Validate alteration screening and outcome-based valuation independently
   before exposing stronger conclusions to users.

## Physical and statistical limits

- Current distant shop images cannot reliably resolve small condition defects.
  Generated detail is not measurement evidence.
- Centering constrains a grade ceiling; it does not establish a complete grade.
- Absolute dimensions require a trustworthy scale reference or calibrated
  distance. Small dimensions alone do not prove trimming.
- Asking prices must remain distinct from sold prices.
- Narrower uncertainty is useful only when independent validation confirms
  that the intervals still cover the true values.

## Repository grounding

- [README](README.md): centering scope, grade bands, and shop-capture limits.
- [Architecture](docs/ARCHITECTURE.md): existing metrology, AR, recognition,
  provenance, learning, and market subsystems.
- [Field capture](docs/FIELD_CAPTURE.md): resolution limits and capture workflow.
- [In-situ learning](docs/IN_SITU_LEARNING.md): per-device confirmation, owner
  review, embedding gates, and the documented prior-feedback concern.
- [Recognition evaluator](eval/recognition_eval.py): existing OCR baseline and
  acceptance-versus-abstention evaluation.
- [Detail capture](cardcenter/detail.py): full-resolution edge strips supporting
  the measurement pipeline.
- [Photogrammetry](cardcenter/photogrammetry.py): reference-based dimensions and
  scale uncertainty.
- [Valuation](cardcenter/valuation.py): grade priors, downside risk, and the
  distinction between asking and sold prices.
