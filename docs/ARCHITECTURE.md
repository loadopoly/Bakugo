# System Architecture: Bakugo (CardCenter)

**Bakugo** (`cardcenter`) is a high-precision computer vision metrology, batch scanning, AR capture, and market analytics engine for collectible trading cards. Unlike superficial grading apps that output arbitrary single numbers, CardCenter treats card assessment as a physical metrology and statistical estimation problem with honest error bars.

---

## 1. System Pipeline Overview

```
                      +-----------------------------+
                      | Raw Image / Video Stream    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 1. Image Quality Triage     |
                      | (Glare, blur, standoff)     |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 2. Outer Quad & Pose Solve  |
                      | (Subpixel line-fit, K-pose) |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 3. Optical Stack Correction |
                      | (Snell stack refraction)    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 4. Subpixel Border Detector |
                      | (Step-anchored edge profiles|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 5. Grader Capability & Bands|
                      | (PSA/BGS/CGC standards)     |
                      +-----------------------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
+-----------------------------+               +-----------------------------+
| Provenance-Firewalled Store |               | Market Analytics & Sync     |
| (SQLite + Cert Verification)|               | (Almgren-Chriss Liquidation)|
+-----------------------------+               +-----------------------------+
```

---

## 2. Core Modules & Subsystems

### A. Computer Vision & Metrology
* **`cardcenter.geometry`**: Quadrilateral detection via perimeter gradient filtering, subpixel line fitting, homography estimation, and perspective rectification.
* **`cardcenter.optics`**: Multi-layer dielectric stack ray-tracing (Snell's Law), in-plane card surface displacement correction $\delta = t \cdot (\tan \theta_1 - \tan \theta_2)$, and iterative apparent-corner solver.
* **`cardcenter.detect`**: Step-anchored boundary detector with directional gradient projections and confidence scoring.
* **`cardcenter.centering`**: Complete measurement pipeline coordinating quad finding, pose recovery, refraction compensation, and border ratio computation with 95% confidence intervals.
* **`cardcenter.capability`**: Multi-tier capability grader classifying cards into `FULL`, `SINGLE_AXIS`, `PARTIAL`, `GEOMETRY_ONLY` (trim/skew detection for full-bleed cards), or `NONE`.

### B. Batch & Live Capture
* **`cardcenter.multicard`**: Multi-card segmentation in single display case photos or video pan frames with halo rejection and spatial deduplication.
* **`cardcenter.capture`**: `RunningRatio` inverse-variance accumulator with $\chi^2 / \text{dof}$ inflation for outlier protection across video frames.
* **`cardcenter.ar` / `cardcenter.perceptopoly`**: Web-based Augmented Reality capture client with continuous tracking and caliper-grade scale calibration.
* **`cardcenter.serve`**: Ultra-lightweight offline HTTP server running on standard library primitives, optimized for Android Termux.

### C. Versioning, Provenance & Connection
* **`cardcenter.versioning`**: SemVer 2.0.0 parser, runtime capability matrix, forward SQLite schema migrations, and upstream GitHub release checker (`https://github.com/PoodlesOfWar/Bakugo`).
* **`cardcenter.connection`**: Client-server sync protocol, deterministic SHA-256 integrity hashing, and the **Contamination Firewall** ensuring only cert-verified grades train models.
* **`cardcenter.store`**: Provenance-preserving SQLite store tracking scans, certified labels, self-reported tags, and marketplace sentiment.

### D. Financial & Catalog Intelligence
* **`cardcenter.catalog`**: Scryfall & Pokémon TCG API integration, ORB visual feature matching, and resolution-gated collector number OCR.
* **`cardcenter.valuation`**: Grade-premium valuation curves and pricing spreads.
* **`cardcenter.execution`**: Almgren-Chriss optimal liquidation execution for card inventory under market impact and arrival-rate feasibility limits.
* **`cardcenter.liquidity`**: Poisson trade-arrival scoring and order book depth analysis.

---

## 3. Data Integrity & The Contamination Firewall

A key architectural principle in Bakugo is preventing circular model contamination:
1. **Certified Ground Truth**: Labels issued by PSA, BGS, or CGC on a physical slab must include a verifiable `cert_number`. Only these labels are exported for model training by default.
2. **Self-Reported & Crowd Votes**: Stored separately with clear provenance; excluded from training exports unless explicit contamination flags are stamped into audit manifests.
3. **Sync Protocol Verification**: `ConnectionManager` automatically inspects incoming sync bundles and quarantines any certified label missing a certification identifier.
