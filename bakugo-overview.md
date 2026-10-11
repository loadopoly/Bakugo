# Bakugo (`cardcenter`) Overview

Bakugo is an objective trading card centering and condition grading metrology system. It measures what can actually be measured from photographs and video frames with honest, physics-grounded error bars, reporting credible grade ranges rather than unearned point grades.

---

## 1. Core Philosophy

1. **Honest error bars over false precision.** When physical resolution or lighting is insufficient to distinguish wear from noise, Bakugo reports uncertainty or refuses to grade rather than fabricating a score.
2. **Separation of measurement uncertainty and standards ambiguity.** If reputable grading companies disagree on standard cutoffs (e.g. 55/45 vs 60/40 centering for PSA 10), Bakugo reports a band spanning both and attributes the spread to standards ambiguity, not camera error.
3. **Centering sets a ceiling, wear decides the grade.** Centering alone is never presented as a grade. A card with 50/50 centering is "PSA not graded" until corners and edges are measured.
4. **Shop-friendly non-destructive evaluation.** Cards in display cases, sleeves, or across counters cannot be handled, flipped, or placed flat on dark mats. All capture coaching directs movement of the phone (zoom, tilt, reposition), never the card.

---

## 2. Metrology Pipeline

### Centering (`cardcenter/centering.py`)
- Detects outer card boundary and inner artwork frame across candidate edge profiles.
- Rectifies perspective and computes opposing border width ratios (L/R, T/B).
- Derives rigorous confidence intervals based on edge sharpness, line fit residual, and resolution.
- Evaluates ratios against strict and lenient published grader thresholds (PSA, BGS, CGC, SGC).

### Condition: Corners, Edges, Surface (`cardcenter/condition.py`, `cardcenter/condition_evidence.py`)
- **Dual Resolution Tiers (v2.25.0):**
  - **Fine tier:** $\ge 6.0\text{ px/mm}$, blur $< 0.45\text{ mm}$ (whitening $< 0.30\text{ mm}$).
  - **Coarse tier (Shop Mode):** $\ge 3.5\text{ px/mm}$, blur $< 0.80\text{ mm}$. Measurements below fine tier are marked `coarse=True` with wider error bounds ($\sigma = \sqrt{(0.6/\text{res})^2 + (0.4\times\text{blur})^2 + 0.03^2}$) and white fibre detection suppressed.
- **Corners:** Fitted against two local straight edges tangent to the nominal 3.0 mm rounded corner arc. Radial rays measure material loss (`loss_mm`) and white fibre depth (`whitening_mm`).
- **Edges:** Normal profiles sampled every 0.5 mm. Nicks are detected relative to a running median cut line; whitening is measured beyond the visible side face (self-calibrated from camera pose).
- **Surface:** Evaluates specular highlights across consecutive views. Thin dark structures appearing under highlights that do not exist in diffuse lighting are identified as scratch candidates.
- **Evidence Pooling:** Accumulates observations across frames. The median estimates geometric defects (loss and nicks); the 30th percentile estimates whitening (since tilt and glare can only increase perceived whiteness).

### Shared Photometrology Contract (`cardcenter/photometrology.py`)
- Standardized `holon.metrology/1` contract shared across the Holon ecosystem (Perceptopoly, StockOpoly, GeoGraph OCR).
- Fuses scale sources (measured, recorded, projected) in log-space with Cramér–Rao and Planck/quantum resolution floors.
- Standardizes defect measurement and aspect definitions across applications while keeping grade interpretation within Bakugo.

---

## 3. Grade Composition (`cardcenter/grading.py`)

- **Probabilistic Enumeration:** Generates exact discrete probability distributions over integer grades for each aspect based on official grader rubrics.
- **Partial Evidence & Upper Bounds (v2.25.0):** When only some corners or edges are resolvable, unread aspects are assumed pristine (`assessed="partial"`), establishing a strict mathematical upper bound (`AspectGrade.is_partial`, `AspectGrade.upper`).
- **Wear Ceiling (`wear_ceiling`, v2.25.0):** If observable wear on even one corner or edge caps the card below its centering ceiling, Bakugo reports a wear ceiling (e.g. `≤PSA 7 (partial)` or `PSA <=7 (partial)`).
- **Credible Ranges (`range`, v2.25.0):** Replaces low-probability modal guesses with an 80% credible range (e.g. `PSA 8–10`, `PSA 8-10 max`, or `PSA 8–10 (surface ?)`).

---

## 4. Accessibility & UI (v2.25.0)

Designed for field readability in card shops and full WCAG 2.1 compliance:
- **Unblocked Viewport Zoom (WCAG 1.4.4):** Viewport meta allows pinch-to-zoom (`user-scalable=no` removed).
- **Fluid rem Typography:** All fonts defined in `rem` relative to root (`html { font-size: 100% }`), starting at 1rem (16px) with nothing below 0.875rem (14px). Monospace typography is restricted to tabular numbers.
- **High-Contrast Solid HUD:** HUD guidance and aspect status cards are placed in a solid high-contrast panel (`alpha 0.96`) below the video viewfinder, preventing background bleed.
- **AAA 7:1 Contrast:** Color palette strictly satisfies AAA contrast across all light and dark text pairings (`--dim: #B8C4D0`).
- **Non-Color Status Signals:** Chips always combine explicit text, symbols, and status (e.g., `Corners 9 ✓`, `Corners ≤7 ⚠`, `Edges — not read`).
- **Screen Reader Support:** Live guidance and results use `role="status"` and `aria-live="polite"`, semantic `<h2>` section headers, `<th scope="row">` table rows, explicit `aria-label` tags, and throttled live announcements.
- **Ergonomics & Controls:** $\ge 44\times 44\text{ px}$ tap targets, 3px `:focus-visible` rings, reduced motion support, and an interactive `A / A+ / A++` text size switcher.

---

## 5. Deployment & Runtime

- **Web / Mobile Browser:** Served via `cardcenter --serve` (`serve.py`), running entirely locally or behind reverse proxies.
- **Docker Container:** Runs in the `bakugo-app` container in the `hideout` compose stack behind the `bakugo` Caddy proxy with a read-only root filesystem.
- **Android Offline (Termux):** Native Python 3, NumPy, and OpenCV running directly on the handset with zero cloud dependencies.
