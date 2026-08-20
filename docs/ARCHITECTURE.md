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
* **`cardcenter.information`**: Pixel-space Fisher / Cramér–Rao floor plus the QUIPU temporal-spatial rhythm. Boost scales effective independent-row count for fusion only; the single-shot CRB is unchanged.
* **`cardcenter.capture`**: `RunningRatio` inverse-variance accumulator with $\chi^2 / \text{dof}$ inflation. `LiveSession` stores the last `ChannelConditions` and inflates the combined PDG bar when `rhythm_boost < 1`; boost above 1 is a status signal only.
* **`cardcenter.ar` / `cardcenter.perceptopoly`**: Web-based Augmented Reality capture client with continuous tracking and caliper-grade scale calibration.
* **`cardcenter.serve`**: Ultra-lightweight offline HTTP server running on standard library primitives, optimized for Android Termux.

### C. Versioning, Provenance & Connection
* **`cardcenter.versioning`**: SemVer 2.0.0 parser, runtime capability matrix, forward SQLite schema migrations, and upstream GitHub release checker (`https://github.com/PoodlesOfWar/Bakugo`).
* **`cardcenter.connection`**: Client-server sync protocol, deterministic SHA-256 integrity hashing, and the **Contamination Firewall** ensuring only cert-verified grades train models.
* **`cardcenter.store`**: Provenance-preserving SQLite store tracking scans, certified labels, self-reported tags, and marketplace sentiment. Local source of truth; `synced_at` marks a successful cloud mirror.
* **`cardcenter.cloud`**: Optional metadata upsert to the same Supabase *project* as Loadopoly-OCR (`bakugo_scans` / `bakugo_labels`). Anon key only. Photos stay local. Not OCR documents — dedicated tables keep the contamination firewall.
* **`cardcenter.learning`**: Closed-form conjugate learners (OCR confusion, encounter prior, Almgren-Chriss impact, **issued-grade outcomes**). Grade predictions expand only from certified labels; identity reduction keeps the published-table heuristic when the model is empty.
* **`cardcenter.quipu_client`**: Optional Observer link (`CARDCENTER_QUIPU_URL`). Structured observations go up; numeric lexicon priors come down. Best-effort, stdlib-only, never required for a measurement.

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
4. **Grade-outcome expansion**: After a successful import, `ingest_certified_labels` rebuilds `GradeOutcomeModel` from the certified export only. Marketplace votes, self-reports, and this system's own predictions never reach `observe`. Ratio bands do not pool: off-centre 8s cannot pull a gem-centred card.

---

## 4. System Dynamics & Tri-Repo Mesh Integration

Bakugo is formalised as the **Touch Sense Axis** ($\text{axis}=1$) of the 7-D unified learning mesh. See [SYSTEM_DYNAMICS.md](./SYSTEM_DYNAMICS.md) for complete mathematical formulations, closed-loop observer dynamics with QUIPU, and Supabase PostgREST synchronization topology.
