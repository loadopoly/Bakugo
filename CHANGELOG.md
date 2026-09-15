# Changelog

All notable changes to **Bakugo** (`cardcenter`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [2.13.1] - 2026-09-15

### Fixed
- **Live AR never ran recognition**: the capture button posted only to `/measure`, and the
  `/ar/push` stream is downscaled to 540 px, where collector-number glyphs are ~4-5 px
  (below the 12 px `MIN_PX_TO_READ` floor). The full-resolution still is now also posted to
  `/identify` in parallel and the result is shown under the measurement, including when
  `/measure` refuses the frame. Camera request raised to 1920x1080 ideal; the tracking
  stream stays at 540 px.

## [2.13.0] - 2026-09-14

### Added
- **Binder-photo OCR/VLM ingestion tool (`ingest/`)**: A standalone offline batch CLI
  (`python -m ingest.binder_ingest --root <folder> --out <dir>`) that turns a folder of
  sequentially-photographed trading-card binder pages into per-card identity + price
  records, implementing Adam's "OCR/VLM Ingestion -- Trading Card Binder Photo Set" work
  instruction (v2.0, 2026-09-14) end to end: sorts `PXL_YYYYMMDD_xxxxxxxxx`-style filenames
  by numeric tail, pairs consecutive front/back page photos, segments each into a configured
  pocket grid (default 3x3), and maps a front pocket to its physically-mirrored back pocket
  via `back_column = (cols + 1) - front_column` (the page flips about its left/spine edge, so
  columns reverse and rows don't). `ingest/binder_grid.py` finds the PAGE's own outer quad and
  rectifies just that rather than trying to detect each pocket's own edges: tested against
  real binder photos, a card-vs-background quad finder run on a whole page reliably finds the
  page's outer edge, not the nine inner ones, because adjacent pockets share a faint plastic
  seam rather than a contrast boundary. Deliberately does not import `cardcenter`'s own
  card-quad detector (tuned for a single card against background, not a 3x3 sheet of them) or
  its OCR preprocessing (tuned for digit-only collector numbers, not label text) -- it is
  self-contained by design, per its own module docstring, and has no runtime dependency on
  `cardcenter` at all.
- **`ingest/binder_sticker.py`**: Rotation-searching OCR (0/90/180/270 per crop, scored by
  regex-plausibility) over a rectified pocket, plus HSV-hue card-back franchise classification
  for the Work Instruction's mixed-franchise-page check. Rectifying near the source photo's own
  resolution (`rectify_page`'s `target_px_per_source_px`) rather than to a fixed canvas fixed a
  total OCR failure found during development: a downscale-then-crop chain lost enough real
  pixels from a sticker's small print that Tesseract read pure noise, while the identical crop
  re-rectified near-native resolution read cleanly. `price_usd` is set only from a literal
  `$X.XX` regex match on the winning OCR text and is never inferred, defaulted, or carried over
  from a neighbouring pocket -- the same report-ambiguity-rather-than-guess posture already
  used by `cardcenter/ocr.py` and `cardcenter/catalog.py`.
- Every exception path in the Work Instruction (S8) is a distinct status rather than a dropped
  record: `no_card_detected`, `pending_price` (unpaired trailing front page),
  `price_needs_review`, `pocket_mismatch` (one side of a mirrored pair looks occupied, the
  other empty), `grid_mismatch` (front/back rectified aspect ratios disagree by more than 15%,
  the closest available proxy for "these two photos are not really the same physical page"),
  and `franchise_mismatch` (a back design's classified hue confidently disagrees with its
  containing folder). Verified end to end against a real page-pair set: the front-to-back
  column mirroring is confirmed correct on real photos, and the `grid_mismatch` guard correctly
  refused a synthetic, non-corresponding front/back pairing built specifically to test it
  rather than silently processing it.

### Known limits
Requires `pip install pytesseract` plus the Tesseract OCR binary on PATH; neither was present
on the plain Windows dev checkout this was built on (page/grid detection has no OCR dependency
and works without them, but identity and sticker text do not). Price-digit OCR (`$X.XX`) did
not successfully extract on any tested real sticker as of this build, so expect
`price_needs_review` more often than ideal until this is revisited with more real stickers to
tune against. `read_front_identity`'s raw OCR of card-front text came back mostly noise in the
one full-pipeline run tested (9 of 9 occupied pockets) -- treat `card_identity_ocr` as a hint
for a human, not usable text on its own, in this build. `classify_back_franchise`'s Yugioh
reference hue is not calibrated against a real sample (none was available in the source photo
set) and is explicitly flagged in-code as less trustworthy than the Pokemon reference, which
was measured from real photos; a real test run produced one plausible false-positive
`franchise_mismatch` at a confidence just over the reporting threshold, consistent with that
caveat. Grid dimensions are configured (`--rows`/`--cols`, default 3x3), not auto-detected per
page -- two different auto-detection approaches (edge-projection seam profiling, per-pocket
quad detection) were tried and failed on real photos; see `binder_grid.py`'s module docstring
for the specific measured failure modes. Per Adam's own closing note on the Work Instruction,
every price/identity/group value this tool emits is meant to be spot-checked by a human before
being treated as authoritative.

## [2.12.0] - 2026-09-14

### Added
- **Interstitial-margin confidence gate (`cardcenter/confidence.py`)**: An auto-accept /
  review / reject gate for the grade decision, keyed to the shot-noise information floor
  already computed in `information.py` rather than a raw score margin. It reads
  `cramer_rao_ratio_pp` (the Cramér-Rao lower bound on the centering ratio) and
  `shot_noise_consistency`, and fires "accept" only when the measurement sits `k_sigma`
  (default 3) CR-sigmas clear of the boundary on a photon-limited frame. `max_shot_ratio`
  is taken directly from `information.py`'s own shot-noise-limited band; multi-frame fusion
  enters only through the effective independent-row count, never by multiplying Fisher
  information. `gate_from_channel(...)` plugs into the per-frame `ChannelConditions` /
  `SensorModel` the AR loop already produces.
- **Standalone recognition evaluator (`eval/recognition_eval.py`)**: An offline harness that
  imports `cardcenter` read-only and measures the recognition path — Tesseract OCR + species
  snap + collector-number read — against a labeled set. Reports the accept/abstain curve
  ("auto-accepts N% at M% accuracy"), the QUIPU-prior contribution (enabled vs disabled
  delta), and every metric sliced by capture resolution (px/mm) to expose the AR distance
  dependence. Confirms the current path is OCR-only (no VLM, by design) and is built to
  measure a VLM/embedding recognizer against the Tesseract baseline if one is added.

## [2.11.0] - 2026-09-08

### Added
- **Multi-channel chromatic edge gradients (`compute_edge_gradient`)**: Grayscale conversion
  (`0.299R + 0.587G + 0.114B`) previously wiped out edge steps when light yellow, silver, or
  white card borders lay on light pine wood desks, tablecloths, or quartz countertops where the
  luminance contrast was near zero. The detector and tracker now compute multi-channel color
  gradients (`max(Sobel_R, Sobel_G, Sobel_B)`), preserving strong edge responses on iso-luminant
  chromatic transitions.
- **Dual-scale adaptive thresholding in contour gathering**: Replaced the single adaptive
  threshold block with dual-scale passes (fine ~21px for subtle card borders against textured
  backgrounds; coarse ~51px for non-uniform ambient light and shadows), eliminating contour
  starvation on low-contrast cards.
- **Handheld perspective foreshortening expansion (`0.92 < aspect < 1.88`)**: Cards photographed
  under handheld camera tilts (35°–45°) foreshorten along the optical axis, compressing apparent
  aspect ratio to `1.40 * cos(40°) ≈ 1.07`. The previous floor (`aspect > 1.15`) rejected these
  valid angled views. The window is now expanded to `0.92 < aspect < 1.88`, allowing natural handheld
  viewing angles up to ~48° tilt while still excluding elongated caliper beams and tools.
- **Temporal 1€ (One Euro) Filter for subpixel quad corner smoothing**: Integrated `LowPassFilter`
  and `OneEuroFilter` in `cardcenter/ar.py`. Stationary jitter from sensor noise is damped by >60%
  without introducing lag during rapid camera movement.
- **Actionable live guidance banner (`#hud-guidance`)**: Real-time coaching banner on the live AR HUD
  surfacing camera positioning tips (e.g. angle tilt, distance, glare, alignment, multi-view accumulation).
- **Holographic cybernetic card viewfinder template**: 60fps holographic reticle rendering a 2.5" × 3.5"
  card outline with cybernetic corner brackets, pulsing crosshairs, and holographic aura when searching.
- **Hardware gyroscope horizon / spirit level**: Tracks device pitch and roll via `DeviceOrientationEvent`,
  displaying real-time tilt degree in `#hud-spirit-level` and indicating planar alignment.
- **Laser caliper sweep animation**: Renders a glowing laser caliper sweep line across the tracked card
  surface that intensifies during SPRT multi-view convergence and locks upon decision.
- **Peak-sharpness auto-capture on settlement**: Automatically freezes the stabilized AR frame and executes
  high-res metrology upon SPRT settlement with debounce protection.

## [2.10.2] - 2026-09-08

### Fixed
- **Live AR tracker locked onto digital pillarbox / letterbox webcam flanks**: When using a
  smartphone camera (such as Pixel 10 Pro) as a webcam via Windows Connected Camera or UVC in
  portrait mode, the driver delivers a 16:9 landscape video stream with solid black pillarbox
  bars on the left and right flanks. This corrupted global frame statistics in `_gather_contours`
  (`np.median = 0`, forcing Canny to `(0, 0)`), and the high-contrast 1.25-aspect ratio boundaries
  were accepted by `quad_candidates`. Because the center reticle sat within this region,
  `find_card_quad` prioritized the ~164k px² viewport over the true card, and `track_quad` in
  `ARSession` remained permanently glued to the stationary black bars with SPRT frozen at `UNDECIDED`.
- **Sensor/viewport boundary clipping**: `quad_candidates` now includes `touches_frame_boundary`,
  rejecting candidate quads whose edges run along or touch the image/viewport boundaries, or that
  span the full height or width of the sensor. A collectible card being measured must have all four
  edges visible in frame; boundary-coincident edges are artifacts or clipped objects.
- **AR tracker drift guard**: `track_quad` in `cardcenter/ar.py` now explicitly rejects tracking if
  the quad touches or drifts to the sensor boundary, raising `DetectionError` and forcing clean
  re-detection of the card under the reticle.

### Added
- **Active viewport detection (`detect_active_viewport`)**: Automatically detects digital black
  pillarbox (flanks) or letterbox (top/bottom) padding in camera frames while ignoring outer 1px UI
  border lines. `find_card_quad` now isolates the active camera feed when padding is detected,
  ensuring Otsu thresholding and edge detection operate on the true scene without statistical
  corruption, and cleanly translates detected coordinates back into the full stream space.
- **Unit & regression test suite**: Added tests in `tests/test_geometry.py` and `tests/test_ar.py`
  covering clean images, pillarboxed/letterboxed viewport detection, 1px border tolerance,
  boundary-touching quad rejection, and end-to-end `ARSession` tracking in pillarboxed streams.

## [2.10.1] - 2026-08-25

### Fixed
- **`corroborated` claimed a cross-check that never happened**: `Recognition.dex` falls
  back to the matched species' catalogue number when none is printed on the card, so
  `POST /identify` deriving the flag as `dex is not None` reported
  `corroborated: true` alongside the warning "no printed dex number was read". Caught by
  exercising the live endpoint on real photos. `Recognition` now carries an explicit
  `corroborated`, set only when a dex number was read **off the card** and agreed with
  the name.

### Added
- **Upside-down retry**: `enforce_portrait` fixes a 90° rotation but says nothing about
  which end is up, and a card photographed upside down has a perfectly readable name that
  the engine simply cannot read. `recognise_card` now retries a half turn when the upright
  pass finds nothing, and reports which orientation resolved it. The upright case costs
  nothing extra.

## [2.10.0] - 2026-08-25

Card **identification**, for photographs that cannot be measured. v2.9.0 established
that bulk-bin phone photos do not carry sub-millimetre border precision, so metrology
correctly refuses on them. Identification has a different error budget — reading a name
needs legible glyphs, not a precise boundary — so those same photos are not worthless.

### Added
- **`cardcenter/recognise.py`**: sparse OCR over the whole rectified card, snapped to a
  closed vocabulary of species names. Deliberately does *not* crop a fixed fraction of
  the card: with a halo boundary the layout does not line up, and measured on the corpus
  fixed-crop name bands landed on sleeve and background while whole-card sparse text
  returned real names. Rectifies from **full-resolution** pixels — glyph size is the whole
  game for OCR, the opposite of centering where accuracy is flat above 1200 px.
- **`cardcenter/data/species.json`**: 250-entry seed vocabulary (Generation 1 complete
  plus widely-printed later species), each with its National Pokédex number. A species
  outside the file is *refused*, never guessed, so extending the file only ever adds
  coverage.
- **Dex cross-check**: modern cards print the National Pokédex number next to the species.
  When both are read they must agree; disagreement refuses rather than resolving in
  favour of one. Caught two real cases in the corpus — a card reading `Garbodor` printed
  dex 562 (Garbodor is 569), and one reading `Chespin` printed 651 (Chespin is 650).
- **`--identify` CLI flag** and **`POST /identify`** endpoint, so a photo that `/measure`
  rightly refuses is still worth something.

### Results (153 cards located out of 162 photos)
- **17 identified (11%)**, 5 corroborated by a printed dex number, 2 refused on dex
  conflict. 11 distinct species: Genesect ×3, Sableye ×2, Zacian ×2, Zapdos ×2,
  Ambipom ×2, Goodra, Golbat, Thundurus, Toxtricity, Moltres, Zubat.
- Previously there was **no image→identity path at all**: `catalog.identify()` takes a
  name as an argument, and the docstring assumed an operator typed it in.

### Notes on the safety argument
Sparse OCR over artwork returns mostly garbage, so the closed vocabulary is the entire
defence — the same "closer to exactly one member than to any other" rule
`read_collector_number` already uses. The edit budget is **inverted from the obvious
choice**: it shrinks with token length rather than growing, because the number of garbage
strings within one edit of a target does not shrink. A first version, generous on short
names, reported **"Seel" 22 times out of 58 identifications** from readings of `Sees`,
`seal` and `peel`; none of those cards was a Seel. Short names must now be read exactly.
Lowering the length floor from 5 to 4 was then measured to change nothing (same 17
identifications, no new false positives), confirming the edit budget — not the floor —
was doing the safety work, so four-letter species remain identifiable from an exact read.

### Known limits
The 250-species seed and species-only scope are the binding constraints, not the optics:
Trainer and Energy cards carry no species name and are refused by construction, and the
full National Pokédex is 1025 entries. Coverage scales directly with `species.json`.

## [2.9.0] - 2026-08-25

Detection rebuilt against a 162-photo corpus of **raw cards in penny sleeves,
handheld over bulk bins** — backgrounds made entirely of other cards, sleeve
glare, real perspective. The previous pipeline measured **0 of 162** and reached
the border stage on 59%. It now reaches the border stage on **93%**.

### Added
- **Robust line fitting (`geometry.fit_line_robust`)**: total-least-squares with a
  trimmed-quantile robust seed followed by MAD-based outlier rejection. `fit_line_tls`
  is least-squares, whose breakdown point is zero, and the points assigned to a card
  side in a real photo are a *mixture of surfaces* — the card's cut edge, the penny
  sleeve's edge running parallel 1–3 mm out, sleeve crinkles, and neighbouring cards.
  Plain TLS returned side residuals of 46–68 px on cards whose raw quad was already a
  correct 1.33 aspect. The robust seed matters independently: contamination covering
  only part of a side *tilts* the fit rather than offsetting it, so residuals vary
  smoothly and MAD alone has nothing to separate.
- **Edge snapping (`geometry.snap_quad_to_edges`)**: pulls each side of an approximate
  quad onto the peak gradient along its normal, searching far inward and barely
  outward. Thresholding a card against a bin of other cards yields a **halo** — right
  shape, right place, several millimetres too big. A halo is the dangerous failure: it
  adds the same margin to opposite sides, pulling any ratio toward 50/50 with a tight
  error bar. `edge_support` cannot catch it, because a halo in a bin of cards still
  lies on real intensity steps — just the wrong cards'. Accepted only when the snap
  improves edge support *and* the area change is a plausible shrink, since a collapse
  onto interior artwork can score higher support than the true boundary. Accepted on
  77% of corpus images, shrinking halos ~19% linearly.
- **Subject prior on the still-photo path (`centering.measure_centering`)**: a
  single-card measurement is a photograph *of* a card, so the subject is the large
  object containing the frame centre. Only the AR path had a reticle; stills had no
  notion of subject at all.

### Changed
- **Per-column border reference (`detect._profile_pass`)**: deltaE is now measured
  against each column's own outer band, smoothed along the side, rather than one
  reference colour per side. Silver and foil borders on modern cards are specular, so
  under any directional light the border sweeps bright-to-dark *along* its length;
  against a single global reference every column reads as far from it, inflating noise
  and collapsing signal. This was the single largest cause of "no measurable border":
  **64 → 9** occurrences, on cards with perfectly visible printed frames. Border widths
  now recover a plausible 2.98 mm median (real Pokémon borders are ~3 mm) at 0.19 mm
  standard error with 3% of columns rejected.
- **Aspect is a selection signal, not just a final veto (`geometry.find_card_quad`)**:
  the residual ladder relaxes until something passes, then takes the largest survivor —
  which assumed residual separates cards from non-cards. On handheld photos it does the
  opposite: printed artwork has crisp straight edges while the card's outer boundary
  does not, so a 3.2%-of-frame blob refining to aspect 1.01 repeatedly beat the
  82.6%-of-frame card at 1.49, and the aspect gate then rejected the blob *and the whole
  frame with it*. `aspect_reject`: **58 → 11**.
- **Subject selection is outermost-wins (`geometry.find_card_quad`)**: when a subject
  point is given, among card-shaped candidates containing it the outermost is the card —
  the same nesting fact `multicard.py` already relies on for dedupe. Residual no longer
  chooses; it reports boundary uncertainty. The support floor starts at 0.55 ("a real
  edge") rather than 0.75, because starting stricter recreated the same early-exit trap
  on a different axis.

### Fixed
- **Container guard misfired on ordinary single cards (`geometry.find_card_quad`)**:
  "two or more non-overlapping card-shaped regions inside the winner" is true of a
  display case *and* of any card, whose artwork panel and text box are card-ish
  rectangles. The guard now also requires the nested regions to **cover ≥50% of** the
  container: a tray's cells tile it, a card's internal rectangles merely dot it.
  `container_guard` false positives: **9 → 0**.
- **Refinement sanity check (`geometry._refinement_is_sane`)**: line-intersection
  refinement of near-parallel sides produced refined aspects of 15, 20 and 409 from raw
  quads that were a correct 1.3. Robust fitting makes this *more* likely, not less —
  rejecting hard enough can leave a small spurious run that fits its own line tightly,
  so the residual looks fine and the old residual-only decline never fired. Refinement
  is now judged by how far it moved the quad, and declines to the raw quad instead of
  discarding the candidate.
- **Ungrammatical shadow refusal**: "a at least 4.00mm edge shadow" → "an at-least
  4.00mm edge shadow".

### Known limits
The corpus still yields only 1 reported measurement, and that is the tool working as
designed rather than a remaining bug. With geometry fixed, 92 images now fail on
border-detection *confidence*: the per-column spread of the located border edge is
~1.3 mm on handheld shots through glossy sleeves, so the confidence gate refuses rather
than emit a ratio it cannot support. Bulk-bin phone photos do not carry sub-millimetre
border precision; a controlled shot is still required for a grade-relevant number.

## [2.8.2] - 2026-08-25

### Fixed
- **Multi-card container silently measured as one card (`cardcenter/geometry.py`)**: `find_card_quad` could accept the outer rim of a packed display case, binder page, or several cards pushed edge to edge as if it were a single card's boundary -- that rim is often a strong, straight, plausibly card-aspect rectangle in its own right, so it passed every existing gate, and border detection downstream could even find *some* signal along it (the tray's own bezel, a neighbouring card's edge) and report a confident but meaningless centering ratio and grade estimate. This was worse than refusing: on a 3x3 tray of graded slabs, the live AR HUD reported a narrowing "~PSA 6 (50%)" estimate for a region that was not a card at all. `find_card_quad` now detects when its winning quad contains two or more other already-valid, meaningfully-smaller, mutually-non-overlapping card-shaped candidates -- the signature of a container rather than a card -- and either raises a clear, actionable error (no `prefer_point` given) or drills down to the sibling nearest `prefer_point` (the cell the camera is actually aimed at). `ARSession.push()` now also surfaces this specific message to the live HUD instead of a generic "point at a card" when it fires. Added `tests/test_geometry.py::test_find_card_quad_refuses_a_packed_tray_without_a_reticle` and `::test_find_card_quad_drills_into_the_cell_under_the_reticle`.

## [2.8.1] - 2026-08-23

### Fixed
- **Shadow-width saturation (`cardcenter/illumination.py`)**: `detect_edge_shadow`'s inner-profile scan stopped at a fixed 1.6mm depth, so any dark band deeper than that (normal desk/directional lighting, not just extreme cases) was silently reported as exactly `1.60mm` regardless of its true width — every affected shot showed an identical, coincidence-looking "edge shadow" error. The scan now runs to `SHADOW_SCAN_MAX_DEPTH_MM = 4.0mm` at higher resolution, and `ShadowVerdict.shadow_unbounded` flags when the dark band still hadn't ended at the far edge of the scan, so `centering.py`'s refusal message says "at least Xmm" instead of a falsely-precise number.
- **Candidate selection ignored what the camera was pointed at (`cardcenter/geometry.py`, `cardcenter/ar.py`)**: `find_card_quad` always picked the *largest* card-shaped region surviving the quality gates, which is wrong the moment a real table holds more than one card — a bigger neighbour would silently outrank the card under the reticle. Added an optional `prefer_point` parameter that re-ranks the same gate-surviving candidates by proximity instead of area (including the top-14 pre-refinement cut, so a smaller target isn't discarded before scoring). `ARSession.push()` now passes the frame centre on first acquisition and the last known card position on re-acquisition after a lost track, so the AR loop measures what it was already looking at instead of whatever else is on the table. Single-card scenes are unaffected — ties only matter when more than one candidate survives.

### Changed
- **Live AR grade readout (`cardcenter/ar.py`, `cardcenter/serve.py`)**: `ARSession.push()` now also calls the existing `predict_overall_grade` (previously wired only into the still-photo `/measure` path) to produce a single most-likely grade and a confidence score alongside the honest worst-case ceiling band. The live HUD leads with that estimate (`~PSA 9 (72%)`) and labels the readout `narrowing (N views)` until `ARSession.settled` actually fires, instead of showing a wide first-frame confidence band (e.g. "7-10") with the same visual weight as a converged one.

## [2.8.0] - 2026-08-22

### Added
- **Dual Mobile Native & WebApp Architecture**: Complete cross-platform mobile native project scaffold in `mobile/` using Capacitor with native camera, device info, network, and haptic support.
- **Universal Container Backend (`cardcenter/serve.py`)**:
  - Full Cross-Origin Resource Sharing (CORS) preflight support (`do_OPTIONS()`) and complete access control headers across all endpoints, enabling mobile shells (`capacitor://localhost`) and external IPs to connect without restrictions.
  - Real-time streaming AR endpoints (`POST /ar/session`, `POST /ar/push`, `POST /ar/reset`) providing continuous subpixel quad tracking, frame quality gating, and SPRT boundary convergence.
  - Server discovery and health endpoints (`GET /health`, `GET /config`, `GET /manifest.json`).
- **Live AR & Web Audio Synthesizer Interface**:
  - Live AR video canvas mode with animated perspective targeting quads, laser caliper sweep lines, dynamic centering ratio badges, and SPRT decision dials.
  - Custom Web Audio API Synthesizer generating cybernetic audio cues for edge lock, boundary settlement, and warnings without external media files.
  - Dynamic container host settings modal allowing mobile and web users to configure target container IP / tunnel addresses on the fly.
- **PWA Capabilities**: Full Progressive Web App manifest (`/manifest.json`) and mobile standalone support for zero-install home-screen app launching on iOS and Android.
- **Automated Test Suite (`tests/test_serve_api.py`)**: Comprehensive test suite covering CORS preflights, discovery endpoints, streaming AR pushes, and multi-tenant scoping.

## [2.7.0] - 2026-08-22

### Added
- **Multi-Tenant User Isolation Endpoints (`GET /my-scans`, `GET /my-analytics`)**: External users querying via `bakugo.loadopoly.com` can only view and analyze their own scans scoped by `X-Device-ID`. Raw SQLite and DuckDB internal databases are completely shielded from public query access.
- **Client Device Scoping (`cardcenter/serve.py`)**: Persistent anonymous `deviceId` in browser `localStorage` automatically included in multipart uploads and request headers.
- **Tenant Query Helper (`ScanStore.scans_for_device()`)**: Indexed retrieval on `idx_scans_tenant(device_id, created_at)` with strict zero-leakage fallback.
- **Unit Test Suite (`tests/test_tenant_isolation.py`)**: Tests verifying cross-tenant scan isolation and tenant-scoped DuckDB OLAP aggregations.

## [2.6.0] - 2026-08-22

### Added
- **DuckDB Analytical Layer (`cardcenter/analytics.py`)**: Zero-copy OLAP engine that attaches the live SQLite store via DuckDB's native scanner. Provides `scan_summary()`, `centering_distribution()`, `device_leaderboard()`, `label_provenance()`, contamination-firewalled `training_export()`, and Hive-partitioned `export_parquet()`.
- **SQLite WAL Mode**: Enabled `PRAGMA journal_mode=WAL` in `ScanStore.__init__()` for unlimited concurrent readers during active scans — required for DuckDB zero-lock attachment.
- **Multi-Tenant Device Isolation**: Added `device_id` column and `idx_scans_tenant(device_id, created_at)` index to the scans schema for per-device query scoping when serving external users via `bakugo.loadopoly.com`.
- **Parquet Lakehouse Sink**: `AnalyticsEngine.export_parquet()` writes scans into `/data/parquet/year=YYYY/month=MM/` for lock-free downstream consumption by dashboards and Loadopoly-OCR's DuckDB-WASM browser engine.
- **Optional Dependency**: DuckDB is added under `[project.optional-dependencies] analytics = ["duckdb>=1.0.0"]` — mobile Termux installs remain clean.
- **Unit Test Suite (`tests/test_analytics.py`)**: 12 tests covering summaries, histograms, leaderboards, provenance, training export, Parquet export, and lifecycle.

### Changed
- **Schema version**: Bumped from `cardcenter/2` to `cardcenter/3`.
- **Dockerfile**: Now installs `duckdb>=1.0.0` and creates `/data/parquet` volume.

## [2.5.0] - 2026-08-22

### Added
- **Multi-View Evidence Fusion (`cardcenter/evidence.py`)**: Implements `fuse()` using Particle Data Group (PDG) scale inflation ($\sqrt{\max(1.0, \chi^2/\text{dof})}$) and mandatory consistency gating (`MAX_CONSISTENCY = 3.0`, `GOOD_CONSISTENCY = 1.5`) to eliminate false precision from disagreeing views.
- **Wald's Sequential Probability Ratio Test (SPRT)**: `SequentialBoundaryTest` accumulates log-likelihood ratio drift per view against grading boundaries (e.g. 55/45 PSA 10), enabling mathematically optimal stopping rather than fixed frame counts.
- **Boundary Information Value**: `information_value()` computes Fisher information utility ($e^{-0.5 z^2}$) to determine if additional camera views are mathematically justified.
- **ARSession Multi-View Fusion**: Integrated `self.fusion`, `self.verdict`, `self.settled`, and `self.worth_continuing` into `ARSession` in `cardcenter/ar.py`.
- **Perceptopoly Capability Pre-Routing**: Evaluates single-axis/full capability on close-up cropped slabs before route termination, and returns SPRT verdict and information value metrics.
- **Unit Test Suite (`tests/test_evidence.py`)**: Added 11 focused tests for multi-view fusion, consistency rejection, SPRT stopping, and information values.

## [2.4.0] - 2026-08-20

### Added
- **World Model Grounding (`cardcenter/world_model_grounding.py`)**: Computes physical-space information efficiency ($\eta = \sigma_{CRB}/\sigma$), lossy channel profiles (blur, noise, refraction, glare, quantization), and accumulates physical invariants (holder refractive indices).
- **Physical-Space Observer Feed**: Injects `grounding_annotation` into `quipu_client.py` measurement observations, teaching the QUIPU Observer how lossy real-world physical measurements are.
- **Physical World Summary**: Exposes `physical_world_summary()` to report accumulated priors on information loss and channel degradation.

## [2.3.1] - 2026-08-20

### Added
- **System Dynamics Documentation** (`docs/SYSTEM_DYNAMICS.md`): Documents Bakugo's role as the **Touch Sense Axis** ($\text{axis}=1$) in the Tri-Repo Mesh with QUIPU and Loadopoly-OCR.
- **Bi-directional QUIPU & Supabase Telemetry**: Documented metrology controls, $2\times$ weighted ground-truth feedback reinforcement (`POST /feedback`), and Contamination Firewall enforcement on cloud syncing.
- **Docker Compose Topology Updates**: Containerized configuration enabling automatic Supabase PostgREST mirroring and `host.docker.internal` network routing.

## [2.3.0] - 2026-08-20

### Added
- Live fusion now consumes the QUIPU overlay: `measure_centering` attaches a `CenteringResult.channel`, `LiveSession.observe` stores it, and `worst_ratio` / `settled` / `status` inflate the PDG bar when `rhythm_boost < 1`. Boost above 1 remains a status signal and never tightens the live interval below the inverse-variance combination.
- **QUIPU Observer client** (`cardcenter.quipu_client`): optional `CARDCENTER_QUIPU_URL` feed of structured observations and cached numeric priors. Collector-number OCR may break catalog ties only when a prior dominates by `QUIPU_TIE_DOMINANCE`; otherwise the reading stays ambiguous. `GET /quipu` reports whether the Observer is linked.
- Docker image (`Dockerfile`, `docker-compose.yml`) serving `cardcenter --serve` on `:8765` with optional Observer URL.

### Changed
- `modulate` no longer pins boost at 1.0 when recursive-strengthening potential is zero, so a washed channel can refuse naive `1/sqrt(N)`.

## [2.2.0] - 2026-08-20

### Changed
- Public web app is **Bakugo** at `loadopoly.com/bakugo` (was `/cardcenter`). The Python package and wheel stay `cardcenter`. Old `/cardcenter/` URL redirects.
- Photos stay on-device. Measurement metadata may sync to the Loadopoly-OCR Supabase project when configured.

### Added
- Live AR camera view in the Pages app: rear camera + `ARSession` guidance for glare, standoff, crop, and square-up, so a messy counter is usable without a clean still.
- **Local + cloud persist**: every successful measure writes `ScanStore` (`CARDCENTER_DB` / browser sqlite). Optional metadata upsert to `bakugo_scans` / `bakugo_labels` in the same Supabase project as Loadopoly-OCR (`cardcenter.cloud`, CLI `--sync-cloud`). Anon key only; never a service-role key. Photos are not uploaded.
- **QUIPU temporal-spatial overlay on the pixel-space lossy channel** (`cardcenter.information`): coherence, relational wash, 7-D Weyl centroid, and boost in `[0.5, 1.5]`. Boost scales effective independent-row count for multi-frame fusion only; the single-shot Cramér–Rao bound is unchanged. `audit_measurement` reports the rhythm and a wash-adjusted fusion floor. `LiveSession.rhythm_boost` consumes multi-frame χ²/dof so disagreeing frames refuse naive `1/√N` credit.

### Security
- Cloud writes reuse the contamination firewall: certified labels without a cert number never leave the device. Pages config is `config.json` (gitignored) or `window.__BAKUGO_SUPABASE__`; the committed file is `config.example.json`.

---

## [2.1.0] - 2026-08-19

### Added
- **Expandable grade-outcome model (`cardcenter.learning.GradeOutcomeModel`)**:
  Dirichlet-multinomial posterior over issued grades, stratified by grader, centering-ratio band, and detection quality. Identity reduction: with zero certified observations, `predict_overall_grade` is identical to the published-table heuristic.
- **Certified ingest path**: `ingest_certified_labels` rebuilds the model from `ScanStore.export_training_set()` (certified + cert number only). `ConnectionManager.import_payload` / `sync` persist the rebuilt model so predictions expand as labels arrive. Re-import does not double-count.
- **CLI `--ingest-grades --db`**: rebuild the model from an existing store. Measure and JSON output now load `CARDCENTER_DB` / `--db` and report `used_learned` / `n_observations`.
- Public exports: `GradeOutcomeModel`, `LearningStore`, `ingest_certified_labels`, `maybe_load_grade_model`.

### Changed
- `predict_overall_grade` / `predict_all_grades` accept an optional `model`. Certified mass blends with the heuristic prior (`GRADE_PRIOR_STRENGTH = 8`); the published centering ceiling is never raised. Subgrades stay heuristic.
- Serve / Perceptopoly payloads include `used_learned` and `n_observations` when a persisted model is present.

### Security
- Circularity firewall unchanged: `observe` refuses `MODEL_PREDICTED`, `MARKETPLACE_VOTE`, `SELF_REPORTED`, and certified-without-cert. Training export remains certified-only.

---

## [2.0.0] - 2026-08-18

### Added
- **Comprehensive Card Grading & Subgrade Predictions (`cardcenter.grading`)**:
  - `CardGradePrediction` dataclass providing predicted overall grades (e.g. `PSA 10 Gem Mint`, `BGS 9.5 Gem Mint`), condition tier classification, and 4-subgrade breakdown (Centering, Corners, Edges, Surface).
  - `predict_overall_grade` & `predict_all_grades` computing grade probability distributions and confidence scores across PSA, BGS, CGC, SGC, and ACE.
  - CLI and Web/AR payloads displaying prominent predicted overall grades alongside centering ceilings.

- **Semantic Versioning Engine (`cardcenter.versioning`)**:
  - `SemVer` class implementing full SemVer 2.0.0 parsing, comparison, and compatibility checks.
  - `VersionInfo` reporting detailed environment, platform architecture, and dependency versions.
  - `EngineCapabilities` dynamic feature matrix tracking metrology, AR, OCR, and sync capabilities.
  - `check_for_updates` querying GitHub releases/tags directly from [`https://github.com/PoodlesOfWar/Bakugo`](https://github.com/PoodlesOfWar/Bakugo).
  - Database schema versioning and forward migration engine (`migrate_database`) supporting `cardcenter/1` to `cardcenter/2`.

- **Remote Connection & Synchronization (`cardcenter.connection`)**:
  - `ConnectionManager` and `ConnectionSpec` for managing remote hubs.
  - `SyncPayload` and `SyncResult` supporting distributed bidirectional sync between mobile devices (Termux) and central database vaults.
  - Deterministic SHA-256 integrity checksums for scan records and wire payloads.
  - **Contamination Firewall**: Enforces strict provenance by quarantining incoming certified labels that lack verified physical slab certification numbers.
  - Endpoint health diagnostics measuring round-trip latency and schema compatibility.

- **CLI Enhancements (`cardcenter.cli`)**:
  - `--info`: Outputs detailed build, runtime, and schema versioning info.
  - `--capabilities`: Displays active engine capabilities.
  - `--check-updates`: Probes upstream GitHub repository for new releases.
  - `--check-health <URL>`: Checks remote endpoint health and latency.
  - `--sync-url <URL> --db <PATH>`: Synchronizes local SQLite store with a remote hub.
  - `--migrate-db <PATH>`: Migrates SQLite database schemas to `cardcenter/2`.

- **Comprehensive Documentation Suite (`docs/`)**:
  - `docs/ARCHITECTURE.md`: Pipeline design, optical ray-tracing, and contamination firewall.
  - `docs/GETTING_STARTED.md`: Setup guide for Android (Termux), Linux, macOS, and Windows.
  - `docs/CLI_GUIDE.md`: Full CLI reference and examples.
  - `docs/VERSIONING_AND_CONNECTION.md`: Detailed specification for versioning, migrations, and sync protocols.
  - `docs/API_REFERENCE.md`: Complete Python API reference.
  - `docs/MATHEMATICAL_FOUNDATIONS.md`: First-principles derivations for optical stack refraction, sensor fusion, and Almgren-Chriss liquidation.

- **Automated Test Suites**:
  - `tests/test_versioning.py`: Testing SemVer parsing, comparisons, capability matrix, schema migrations, and update checks.
  - `tests/test_connection.py`: Testing sync protocols, payload serialization, and contamination firewall quarantining.

### Changed
- Refactored `cardcenter/channel.py` `select_by_consensus` to integrate 3D path continuity into consensus selection.
- Enhanced `ScanStore` with `_schema_metadata` table and automated schema version tagging.
- Modernized `README.md` with status badges, upstream repository links, and a complete documentation index.

---

## [1.9.0] - Prior Release
- Initial release with optical stack refraction, multi-card detection, AR capture, and Almgren-Chriss liquidation.
