# Changelog

All notable changes to **Bakugo** (`cardcenter`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [2.21.1] - 2026-09-24

From eight screenshots of 2.21.0 (Meganium, live view). The card filled 50-80% of the view
and every frame was out of focus for about two minutes. Nothing was measured live, and the
one Freeze was refused as "border confidence too low ... even, diffuse lighting". The phone
was too close for the camera to focus, and the app never said so where it could be read.

### Fixed
- **The "too close" advice is readable and says how far.** The banner cut off at "frame is soft
  -- if it stays sof", under the tilt chip. It now wraps, and the tilt chip moved to the top
  right. With the card over 45% of the view and the frame soft, it reads "too close to focus --
  lift the phone until the card is about a third of the screen wide" (about 12-15 cm on a main
  camera, which still gives about 12 px/mm in the Freeze frame).
- **A soft photo is refused for focus, not for the light.** A soft photo fails whichever check
  runs first ("border confidence too low", "edge shadow"), and those messages sent the user to
  change the lighting. The photo's sharpness is now measured on the card at a fixed 8 px/mm:
  155-267 on the field stills that measured, 6-39 on the ones refused. Below 60, the refusal
  says the photo is out of focus and what to do, followed by the original reason.
- **Freeze keeps the sharpest of four frames.** It takes four video frames 110 ms apart and
  sends the sharpest, ranked by focus over the tracked card. The debug line shows the four
  scores.

### Added
- **Server time per push.** Each live push reply carries `server_ms`, and the debug inset
  shows it beside the round trip (`...ms (srv ...)`). The next "PUSH TIMED OUT" can then be
  split into server time and network time.
- **Unmeasured live frames are kept too.** The field log keeps one live frame every 6 s even
  when nothing is measured (`live_unmeasured`), with the tracking state, the guidance and the
  server time. A session like this one then leaves frames to replay.

## [2.21.0] - 2026-09-24

From fourteen screenshots of 2.20.0 ("multi views on Terapagos with no capture", "incorrect
association with Meganium") and, for the first time, the exact photos the server received:
the field log kept every Freeze still and measured live frame, and each result below is from
replaying those files.

### Fixed
- **Meganium was named Seaking.** The name vocabulary was a 250-species seed without
  Meganium or Terapagos, so the best match left was "soaking" in the flavour text, one letter
  from Seaking. The vocabulary is now the whole National Pokedex (1-1025). Names are matched on
  letters only, so two-word names ("Mr. Mime", "Iron Valiant") are read too.
- **A flavour-text word no longer names a card.** OCR now reports where each word sits. A word
  in the name bar beats any word lower on the card; a word lower down counts only if it reads
  exactly as a species; the word after "from" ("Evolves from Bayleef") is skipped. When the whole
  card doesn't give a name, the name bar is read again on its own at twice the size.
- **The name was cut off the card that OCR saw.** On a sleeved holo card the outline's top edge
  sometimes sat on the artwork, below the name bar. The card is now straightened for reading
  with 15% extra above and below it. On the five field stills: Meganium x3 and Terapagos x2 named
  (was one Seaking and four "no name").
- **Sleeved cards measured from the sleeve, the shadow, or the printed frame.** The outline
  often ends up one band off the card. It can sit on the sleeve's seam, with the sleeve's spare
  length and the card's shadow inside it (Terapagos, 3.2 mm). It can sit on the sleeve's margin
  (Meganium, right side, 1.3 mm). Or it can sit on the printed frame with the silver border
  outside it (Meganium, left side, 2.4 mm in). Before measuring, each side is now checked by
  colour. Through clear plastic, the sleeve's margin keeps the counter's colour, and so does the
  card's shadow, only darker. A silver border, the green frame and holo foil don't. Each side is
  moved onto the card's edge where the change of colour is clear along the side.
  - Replaying the stills: the Meganium that read 68.4/31.6 now reads 56.6/43.4 (L 2.46,
    R 2.70, T 2.09, B 2.73 mm). A second Meganium photo that was refused now reads 52.6. A
    Terapagos that was refused ("3.81 mm edge shadow") now reads 57.5, and the live view of it
    settles at 57.9.
  - The 144 synthetic captures are unchanged.
  - A dark mat is left to the previous check.
  - A band found on both opposite sides at the same width is treated as the card's own border,
    not a sleeve.
- **Live pushes piled up behind each other.** The phone stops waiting for a push after 4 s, but
  the server kept working on it, and the next push started alongside it on the same session.
  The container has two CPUs, and the OCR for a Freeze took both. Now a session takes one push
  at a time. If the previous one is still running after 1.5 s the server answers "busy" instead
  of queueing, and the phone shows SERVER CATCHING UP rather than a failure. Tesseract runs on
  one thread at lower priority, so the live view is served first.

### Known
- The outline finder is sensitive to the crop: the same Meganium photo cropped around the card
  outlined the artwork instead. The side check above repairs the cases it can, not that one.
- Two photos of the same Meganium read 56.6 and 52.6 vertically. That is closer than before,
  but not yet as close as two photos of one card should be.

## [2.20.0] - 2026-09-24

From sixteen screenshots of 2.19.1 (penny-sleeved Meganium and Terapagos on a light wood
table, Wi-Fi). Freeze came back "could not locate a card-shaped quadrilateral" and Identify
"no card found here" on cards in plain view, while the live frames of the same seconds found
the card every time (the debug inset showed the exact frames the server got, outlined).

### Changed
- **Freeze / Measure Card uses the video frame when the video is big enough.** The still was
  the camera's separate photo (`ImageCapture.takePhoto`): taken with its own focus and field of
  view, and cropped to the card by mapping the tracked outline from the preview into it -- a
  guess about how the two line up. The phone streams 2160x3840 here, so the video frame
  itself has the card ~1000 px across (~17 px/mm), at the same pixels the outline was found
  in, and it is now the still whenever the video is 3.5 MP or more (the photo is still used
  when the video is small). With no mapping to allow for, the crop around the card is 0.25 of
  its long side each way instead of 0.6. This is the change most likely to fix the refusals;
  the photos themselves were never seen, which is what the next item is for.
- **A live number the photo refused is not shown as settled.** The chip read "66.8% · ~PSA 8 ·
  range 6-10" and the banner "Target settled · Auto-capturing" over "Measurement Refused".
  After a refused photo the chip says "live 66.8% · photo refused, not confirmed" and the
  banner asks for another Freeze, until the live estimate unsettles or a photo is measured.

### Added
- **The server keeps what it measured** (`cardcenter/fieldlog.py`): every still sent to
  /measure and /identify, and the live frames the AR session measured (at most one every two
  seconds), each with a JSON of the outcome (ratio, borders, error, outline, holder, version).
  They go to `field/` beside the database (`/data/field` in the container), newest 200 kept;
  `CARDCENTER_FIELD_LOG=0` turns it off, `CARDCENTER_FIELD_DIR` moves it. Nothing leaves the
  server. Every field report so far had to be rebuilt from screenshots; the next one can be
  replayed from the exact inputs.
- Tests: the still source decision, the unconfirmed chip, the field log (kept byte for byte,
  newest N, off means off, measure and identify both recorded).

### Known limits (seen in this report, not fixed)
- **Holo sheen along a card's top edge can tilt the outline.** On three of the four exact
  frames in the debug inset the server's outline ran from the sleeve's top-left corner down to
  the HP: the rainbow along the top of a holo Meganium is a real straight edge at ~6 degrees,
  and the quad on it is card-shaped at a small tilt. Trying the neighbouring lines for each
  side of the best candidates found other slanted lines, not the level edge, on frames
  rebuilt from the inset (they are 291 px across, blown up), so it was not kept; the exact
  frames the field log now keeps are what that needs.
- **The live Terapagos reading (66.8%) is probably wrong**: the card looks near 50/50 in the
  sharp screenshots, and a frame rebuilt from one measures its bottom border at 1.3 mm where
  the other three are 1.9-2.6 (the holo band along the bottom edge is the suspect). With the photo
  now able to confirm or refuse it, and the frames kept, this is the next thing to look at.

## [2.19.1] - 2026-09-23

### Fixed
- **Measure Card measured the sleeve, not the card.** Checked live after 2.19.0: the same
  Meganium preview through `/measure` still read 77.1 (top border 1.4 mm, bottom 4.6). The
  framing path outlined the penny sleeve itself, ~2.2 mm past the card at the bottom and
  ~1.5 at the top -- beyond the 1.2 mm the 2.19.0 seating looks. Inside a sleeve's margin (the
  mat seen through the plastic: its hue, and less than 0.6 of the way to the card's
  lightness) the card's edge is now looked for as the next step in, up to 3.5 mm, and taken
  only when that step is stronger than the first: the card's edge is the strongest step in
  the profile, a sleeve's thin edge is weak. (A shaded silver border beside a green frame
  passed the colour test on this frame and failed this one.) Asked whatever the holder
  setting, since a sleeve is easy to forget there. The preview now measures 53.9 through
  Measure Card with the holder on Penny Sleeve or on Raw. On a synthetic sleeve (2.2 mm
  below the card, 1.2 above) the ratio was 6.4-8.5 points out and is within 2.5; the 144
  synthetic raw captures measure exactly as before.

## [2.19.0] - 2026-09-23

From thirteen screenshots of 2.18.0 on a black mat and a wallet. The reader found the card
in every usable frame (seven): run again on them it outlines the card each time. Where the
app's outline was off the card (IoU 0.69-0.86 with the fresh one) it was offset the way the
phone had moved -- one of them is exactly where the card sat in the screenshot a second
earlier. What went wrong was around the reader: a settled number that was wrong, and an
outline that trailed a moving phone.

### Fixed
- **A settled 74.2% (~PSA 6) that was about 54/51.** Meganium in a penny sleeve, sharp and
  level: the outline's top side sat 0.4 mm off the card, on the start of a soft edge (the
  sleeve's open end). Every border is measured from the outline, so the top border's colour
  was sampled on that ramp, the ramp's end was taken for the printed border, and the top
  border came out 0.7 mm where it is ~3. `measure_centering` now seats each side of the
  outline on the card's own edge (`centering.seat_outer_edges`, after the edge-shadow check,
  which still handles a one-sided shadow band and charges its correction as uncertainty): the colour profile
  across the edge, median along the side, and the edge put half way from the background to
  the card past the ramp; a side moves only on a clear step that 60% of the side agrees on,
  inward from 0.12 mm and outward only from 0.25 (on a thin border under heavy blur the
  half-way point slides out on an outline that was right). The measurement says when it
  moved a side. The same preview frame now measures 50.9 / 53.4.
  On 144 synthetic captures (four border layouts including 1.0 and 1.2 mm borders, 0 and 20
  degrees, 0-3 px of blur, dark and light backgrounds) with the outline 0.4-0.5 mm off one
  side, the median error went from 3.6-4.3 points to 1.0-1.3, and captures more than 3
  points out from 64 to 14; with the outline on the card nothing changed (median 1.05 ->
  1.13, max 3.38 both).
  Tried as well and dropped: starting the printed-border search past the end of the ramp.
  It changed nothing on the Meganium frame and doubled the >3-point captures under blur.
- **The outline stays on the card between answers.** On shop LTE an answer arrives 0.5-1.5 s
  after its frame, and in the screenshots the outline sat 40-90 px beside the card, where the
  card had been. The page keeps a 96 px grey copy of each frame it sends and, while the
  outline is up, measures how far the live preview has moved from it (block matching, coarse
  to fine, a few ms; up to 30% of the view; nothing when the view has too little texture or
  the match is at the edge of the search) and moves the outline with it. The still
  capture's crop uses the moved outline too. In a browser test with a fake camera sliding
  over a card and a server answering 0.7 s late, the outline trailed the card by a median
  0.51 card widths without it and 0.03 with it (90th percentile 0.07).

### Added
- Six field frames: five from these screenshots (outlines are the reader's own, checked by
  eye, marked `reader`) and the Meganium preview at screen resolution
  (`tests/fixtures/field/previews/`).
- `tests/test_centering_seat.py`: an outline off the card is seated (within 2.5 points on
  captures that were up to 11 out); an outline on the card is left alone, thin blurred
  borders included; the settled field frame measures near centred.
- `tests/browser/test_ar_motion.py`: the fake-camera latency test above.

### Known limits
- **Modern silver-border cards with a pale face** (Drampa, a colourless Scarlet & Violet
  card): the top border is refused ("border detection confidence too low ... 'top'"). The
  silver border and the grey frame inside it differ by dE 7-10, and the name, HP and stage
  tab break the top border up. The refusal is right; measuring these needs the card's
  layout, not a colour step.
- The five new frames are the live preview read out of screenshots, not the exact
  frames the server got, and the bottom 11% (under the guidance banner) is cut off.

## [2.18.0] - 2026-09-23

From five screenshots of 2.17.0 at a counter (now `tests/fixtures/field/counter_*`). In
every one the app said TRACKING over an outline that was not the card, and in four of them it
was nearly the same outline, over three different cards on two different surfaces: the live
tracker was holding a place on the screen, not a card.

### Fixed
- **The live outline stays on a card, or goes.** The tracker only ever asked "is there an
  edge near where the card was?", and on a soft frame from a moving phone there always is.
  Fed the frames with the outlines 2.17.0 drew, it kept all five (moved 3-5 px) where the
  scene search judged none of them a card. Now every tracked outline is judged against the
  frame (card-shaped through the perspective, edges along its sides that stop at its
  corners, a face unlike its surroundings); one that fails is dropped and the card is found
  afresh, and the card is re-found from scratch at least once a second even when the outline
  passes. With nothing found the HUD says so instead of showing a box.
- **The outline follows the card when the phone moves.** Frames arrive a few times a second,
  and between two of them a hand-held phone moves the card tens of pixels; the edge search
  reaches 14. The session now measures how the whole view moved (phase correlation on a
  half-size grey frame, ~3 ms) and starts the search where the card went. On three field
  frames shifted 25-80 px between pushes, 2.17.0 kept the outline where the card had been
  (IoU 0.79-0.94 with the old position, 0.30-0.60 with the card) -- the screenshots'
  symptom, reproduced; now it stays on the card (0.76-0.96).
- The tracker starts from the last *measured* outline, not the smoothed one on screen: fed its
  own lagged output it searched where the card had been.
- A tap names a card once. The tapped point used to stay the aim for the rest of the session,
  so every later re-find went back to that spot of the screen whatever the phone was now
  pointing at. It now lasts until its card is found (or 3 s).
- A new card replaces the old one's views: when the outline is re-found somewhere else
  entirely (overlap < 0.3), the pooled measurements start over instead of mixing two cards.
- In the live loop the contour detector's outline is used only if the scene search also judges
  it a card (`locate_card(judged_only=True)`); alone it returns shapes on a counter that are
  not cards.

### Added
- **Half a card is not a card.** Too close, the card's bottom runs out of the view, and its top
  border, sides and the bottom edge of the art window make a clean quad that is card-shaped
  lying sideways (three of the five frames). A sideways card whose sides carry on past its
  edge, where standing it upright would take it out of the frame (or to within 6% of its
  edge), is refused with "whole card
  not in view -- lift the phone until all four edges of the card show".
- **The rest of the card is looked for directly.** When the best quad is the top part of a
  card and the card's bottom edge is faint, the scene search now tries the quads that keep
  three sides and move the fourth out to a parallel line, and takes the whole card when it is
  judged a card, the part sits in it the way a card's top half does, the card's sides run on
  along what is added, and what is added is not a card of its own (two cards side by side).
  On busy holo art the whole card was not in the 400-quad shortlist at all.
- "frame is soft -- if it stays soft the phone is too close to focus: lift it a little" when
  the frame is soft and the card is more than half the frame wide (four of the five frames:
  phone still and level, every edge soft).
- Five field frames with outlines and the outline 2.17.0 drew over each (`shown`), and tests:
  a wrong outline is let go; half a card is refused; the outline follows the card through
  25-80 px moves; a tap is used once; the too-close hint.

### Changed
- A tracked frame costs ~45 ms on the server instead of ~25 (judging the outline); a full
  re-find (~250-400 ms here) runs once a second. The scene search now builds its lines only
  when it needs them.

### Known limits
- **Two cards in one holder, offset** (counter_meganium_b: Meganium in front of a Rare Candy
  card that sticks out ~45 px above it): the reader outlines the holder -- the Rare Candy
  card's top edge with Meganium's sides and bottom (IoU 0.72-0.75 with Meganium). Recorded,
  not asserted. (With OpenCV 5, as in the Docker image, it says the whole card is not in view
  instead: in that cropped frame the card's bottom is 12 px from the edge.) As with the stacked pair in 2.17.0, separate the cards to measure one.
- The counter_* frames are the live preview read out of the screenshots (the debug inset was
  hidden), so they are the preview, not the exact server frame, and their bottom 11% was under
  the HUD banner and is cut off. They are all badly out of focus; their outlines are good to
  about +/-6 px and the tests use IoU 0.75 for them.

## [2.17.0] - 2026-09-23

A reader built for the counter, from the owner's own shop-style frames: cards on light
wood at 17-40 degrees of tilt, sleeved, two or three to a frame, one lying on another, a
thumb over an edge, motion blur. Those frames are now the regression set
(`tests/fixtures/field`, with outlines in `annotations.json`); on them the contour detector
alone found the card under the reticle in 0 of 17 checks.

### Added
- **`cardcenter/scene.py`: find cards from straight edges, not closed contours.** A silver
  or sleeve edge on light wood is the weakest edge in the picture, weaker than the artwork
  inside it, so no threshold gives a closed outline. The scene search takes line segments
  (LSD on the L, a and b channels -- the edge is faint in brightness and clear in colour),
  pairs roughly parallel lines around the aim point, and judges every quad on the picture:
  * edge support with one polarity all the way along each side (texture crossing a line has
    random polarity; a boundary does not), three sides well seen and the fourth at least
    partly, so a thumb or an overlapping card over one side is survivable;
  * card shape through the perspective: the quad is un-projected and must be a 63.5 x 88.9
    mm rectangle, so a card at 40 degrees (1.1 in the image) is accepted and a 1.1
    rectangle seen face-on is not;
  * corners where the edges stop: a quad whose two sides both run on past the ends of the
    third is half a card (the artwork's lower edge between the card's own sides) and is
    refused;
  * a face that is not the counter: lines from the counter's edge, one card's side and
    another's end can close a big, well-supported quad around bare wood; if most of its middle
    matches what lies just outside any one of its sides, it is refused;
  * nesting: the sleeve over the card and the card over its printed frame win when nearly as
    well supported, and the card around its art window wins when the window sits in it the
    way card art does (inside the border, starting under the name bar, stopping about half
    way).
  Reports per-side support, whether all four sides were seen, the recovered tilt and aspect.
  On the field frames, end to end the way the phone does it (JPEG at 0.75 and 0.92, a tap on
  the card, two pushes to the live session): 9 of 9 tap points on every card but one (below),
  against 0 before. At the full frame size, 156 of 168 aim points (12 per card, JPEG and PNG);
  every measurable card found by `locate_card` (IoU >= 0.8 with the hand-drawn outline, 0.7
  for the softest); the nearest complete card when the reticle is on bare counter. On synthetic cards on a light background its corners
  are 1-2 px out on a 1440 px frame where the contour detector's were 3-23 px.
- **`locate_card` runs both detectors.** The contour detector's outline is kept when the scene
  search judges it a card about as well supported and it is the same card (it is sub-pixel on a
  plain mat); otherwise the scene search's is used. The live tracker, `/measure` (via
  `framing`) and `/identify` all acquire through it; the live measurement now locates inside its
  crop with it too and passes the outline on, instead of re-running the contour detector alone.
- **Tap a card to pick it.** On a counter the card under the reticle is not always the one
  you want. A tap on the preview sends the point with the next frame (`aim_x`, `aim_y` on
  `/ar/push`, 0..1 of the pushed frame); the session starts over on the card there
  (`ARSession.select`). The same tap still sets the focus point.
- **"Too far" says whether a photo would do.** The live frame is 540 px across, which at a
  counter is almost always too coarse to measure live, while the Measure Card photo has
  the camera's full resolution. The phone now sends `source_scale` (camera pixels per pushed
  pixel) and the guidance says "live view too coarse here (1.5 px/mm) -- hold still and tap
  Measure Card (photo ~6.0 px/mm)", or that even the photo would be too coarse.
- `tests/test_field.py` (the field frames, the tap, the photo hint, synthetic corner accuracy
  on dark and light backgrounds at 0-45 degrees, pose recovery) and a browser test for the
  tap and `source_scale`.

### Changed
- When no card is found the guidance says "no card found here -- point the centre of the view
  at a card, or tap the card on screen" instead of the contour detector's "shoot the card
  against a plain contrasting background", which nobody can do at a shop.

### Known limits
- **A card lying exactly on another, sides aligned** (desk_spread's energy card on the card
  under it): at the live 540 px size the pair is a card-shaped quad better supported than the
  top card alone, and the reader outlines the pair (0 of 9 tap points; 9-11 of 12 at the full
  frame size). Tried and dropped: the phone's tilt as a prior (the pair is card-shaped at a
  plausible tilt too), and refusing quads cut across by a full-width edge (card texture
  triggers it on single cards). Nudging the top card off the one beneath fixes it. The debug
  inset shows the exact frame the server saw: a screenshot of a miss is a new fixture.
- The phone's tilt prior (`SceneSearch(expected_tilt_deg=...)`) is implemented but not sent:
  on the field frames it found the card from 153 of 168 aim points against 156 without.
- Acquisition costs 150-700 ms on a 540 px frame (tracking after that is ~20 ms).

## [2.16.1] - 2026-09-23

From three field screenshots (Android Chrome, 2160x3840 camera, cards on a desk at 17-40
degrees of tilt): the live view said PUSH FAILED on every frame, with the inset reading
`HTTP 200 not JSON: {"ok": true, "trackin...`.

### Fixed
- **`/ar/push` answers were not JSON on soft frames.** When a side of the card has no finite
  information floor (not visible against the background, or too blurred), its capture advice
  carries `gain = inf`, and `json.dumps` wrote that as the bare token `Infinity`. Browsers'
  `JSON.parse` rejects it, so every answer failed to parse while the server was tracking.
  Reproduced on a synthetic 540x632 frame with a 1.5 px blur. Fixed at the source
  (`QuadInformation.to_dict`, `SideInformation.to_dict` write non-finite values as `null`) and
  for every response: `serve._dumps` replaces any non-finite float with `null` and writes with
  `allow_nan=False`, so a future inf fails loudly in tests instead of silently on the phone.
- **The page tolerates it from an older server:** `parseReply` retries with bare
  `NaN`/`Infinity` read as `null` before reporting a parse failure.

### Added
- **Continuous autofocus and tap to focus.** The screenshots were soft close up and the
  guidance was "hold steadier or let it refocus". The track is asked for
  `focusMode: continuous` where it offers it, and a tap on the preview sets the focus point
  (`pointsOfInterest`, single-shot, back to continuous after 4 s). Cameras without focus
  controls ignore the tap.
- Tests: strict-JSON parse of `/ar/push` on a soft frame and of `_dumps` (Python); an older
  server's `Infinity` still tracks, and a tap on a camera without focus controls is a no-op
  (browser).

## [2.16.0] - 2026-09-23

Shop-floor readiness: what broke, or would have, on a phone at a local card shop.

### Changed
- **The still upload is the card, not the photo.** A full-resolution still (3-12 MB) went up
  twice, to `/measure` and `/identify`, with no time limit, over whatever signal the shop has.
  The app now crops the photo to the card it was tracking, with a margin of 0.6x the card's long
  side each way so a small preview-to-photo mapping error cannot cut the card off, and sends
  `crop_x`/`crop_y`/`full_w`/`full_h` so the server keeps the lens geometry: the focal length
  comes from the whole photo's width and the principal point from its centre, the same rule
  `framing` already applied to its own crops. Measured on a 650 mm synthetic capture the crop
  is under a quarter of the whole JPEG and the ratio agrees with the whole-photo measurement to
  within 0.6 pp. No tracked card (gallery upload, lost lock): the photo goes as taken, shrunk
  only if it is over the server's 40 MP / 12 MB limits.
- **Uploads have a deadline.** `/measure` and `/identify` give up after 60 s with a message
  about the signal, instead of leaving the button on "Calculating" for good. The live loop
  pauses while a still is uploading so the two do not share a weak uplink.
- **The live box no longer flickers on a slow link.** It was dropped as stale 800 ms after the
  last answer; at a second per round trip that is every frame. The limit now follows the
  measured round trip (2.5x its running average, 800 ms to 4 s).
- **Card detection floor follows pixels, not frame fraction.** `find_card_quad` ignored regions
  under 0.8% of the frame, so the same card at the same distance was found on a 12 MP phone and
  missed on a 50 MP one. The limit is now 80% of a card at the measurable floor (4.5 px/mm),
  clamped to 0.15-0.8% of the frame (`framing.min_area_frac_for`). On a 27 MP synthetic frame
  with the card at 5 px/mm the old limit locked onto the wrong region (7.2 px/mm reported);
  the new one finds the card and measures within 1 pp.
- **`/identify` locates coarse to fine.** It searched a 1400 px copy of the whole photo, where a
  card across a counter is too small to find or read; it now uses the same two-pass search as
  `/measure` and reads the card inside a generous crop of the original pixels. Quad and
  `image_size` stay in full-photo coordinates.

- **`deploy/Rebuild.ps1` deploys the stack that is actually serving.** On the hideout PC the
  live server is `bakugo-app` in the parent hideout compose project (which builds `./Bakugo`);
  the script ran this repo's compose file, collided with that stack's `bakugo` container name
  and deployed nothing, so the phone kept getting 2.14.0. It now finds the running `bakugo:dev`
  container, reads the compose file and service that own it, and rebuilds and restarts only
  that service (`--no-deps`). `-ComposeFile` / `-Service` override it.

### Added
- Screen wake lock while the camera is up, re-taken when the page becomes visible again.
- Tests: `tests/test_framing.py` (big-sensor floor, phone-side crop intrinsics, crop placement
  validation, crop vs whole-photo agreement, identify at distance) and two browser tests (the
  still upload is the tracked card with its placement, the live loop yields to it; a hung
  upload reports itself).

## [2.15.0] - 2026-09-22

Shooting from across a counter: a shop, a stand, a display case.

### Added
- **`cardcenter/framing.py`: crop to the card, then cap.** Both measurement paths capped the
  whole frame before measuring (2400 px for a photo, 1200 px live), which on a card
  photographed from a distance discarded exactly the pixels the borders needed. The card is
  located first -- coarse to fine, and across two detector working sizes, because detection is
  not monotonic in scale (on a 3000x4000 frame the detector returned the card at 2400 px and
  the artwork panel at 1400 px) -- and the cap then applies to the crop. Intrinsics travel with
  it: cropping does not change the lens, so the focal length scales with the resize only and
  the principal point moves by the crop origin. Measured against ground truth at 313 px of card
  width: 0.08 pp with the crop, 4.34 pp (or a refusal) without.
- **A resolution floor on the photo path.** Below `capture.MIN_PX_PER_MM` (4.5) `/measure`
  refuses and says how many px/mm the card has, instead of reporting a number: at 4.0 px/mm the
  measurement came back 25 pp from truth with an error bar that did not cover the miss.
- **Zoom bar in the live view.** Uses the camera's own zoom where the track exposes one
  (`2.0x cam`, real detail) and falls back to a capture crop otherwise (`2.0x crop`, which
  spends the 540 px detector budget on a smaller region without adding detail).
- **Camera picker** (Settings -> Camera). A telephoto module reaches further than any crop; the
  existing `Lens` dropdown only tells the optics model which field of view to assume and never
  switched cameras.
- **Full-resolution stills.** The measurement photo comes from `ImageCapture.takePhoto()` where
  supported -- the sensor's frame rather than the ~1080p preview -- falling back to the video
  frame, with the inset footer saying which was used. The stream is now requested at up to 4K.
- **`docs/FIELD_CAPTURE.md`**: the px/mm table, what the zoom readout means, and what to do at
  a counter.

### Changed
- The live loop measures a crop around the tracked card rather than the downscaled frame, and
  re-detects inside that crop instead of reusing the tracked outline: the 1-euro filtered quad
  measured up to 4.3 pp from truth where a fresh detection in the crop was within 0.7 pp.
- The "too far away" gate now judges the resolution the crop will have, not the whole frame, so
  shots the crop can measure are no longer refused.

## [2.14.2] - 2026-09-22

### Fixed
- **One hung `/ar/push` killed the session.** The fetch had no timeout, so a request that never
  settled left `isPushing` true for good: every later tick returned early, and the HUD and the
  debug inset froze on the last good response while the camera carried on. Seen in the field as
  a teal box sitting over a scene it no longer described, with an inset frame several seconds
  old. Requests now abort after 4 s (counted as `timeouts`), the watchdog clears a request still
  outstanding after 8 s, and both report the reason.
- **A stale overlay is no longer drawn.** If the last good response is more than 800 ms old the
  quad is dropped and the chip says `RECONNECTING`, instead of leaving a box from an old frame
  on a live preview. The inset footer shows the age of the frame it is displaying (`live` or
  `N.Ns old`).
- **Tracking detail on portrait cameras.** 2.14.1 scaled the cover-cropped region so its LONG
  side was 540, which on a portrait stream (1080x1920 here) made the card smaller than the
  uncropped frame had. The crop is now scaled to 540 across, with the height capped at 960, so
  detail is unchanged on a portrait camera and about 2x better on a landscape one, in both
  cases without the scene the user cannot see.

### Added
- `tests/browser/test_ar_loop.py` covers the two new cases: a request that never answers does
  not stop the loop, and a stale overlay is dropped.

## [2.14.1] - 2026-09-20

### Fixed
- **Live AR could preview forever without ever sending a frame.** The push loop was started
  from a single `loadedmetadata` handler, and every failure inside `arTick` was swallowed by an
  empty `catch`, so a loop that never started and a loop whose every request failed both looked
  like "SEARCHING" with an unpainted 300x150 debug inset. The loop now starts from
  `loadedmetadata`, `playing`, `resize`, an explicit call and a watchdog that retries twice a
  second, all idempotent; failures land in `arStats` (pushes, ok, fail, HTTP status, round-trip,
  frame size, last error), on the HUD chip (`NO CAMERA FRAMES`, `LOOP NOT RUNNING`,
  `FRAME CAPTURE FAILED`, `PUSH FAILED`, `SERVER REFUSED FRAME`), in the inset's footer, and in
  the console. A non-JSON reply (a tunnel or proxy error page) is reported with its status and
  first bytes instead of being discarded.
- **The detector was sent the whole camera frame, not the part the user framed.** `#ar-video`
  is `object-fit: cover`, so a 1920x1080 stream in a portrait viewport shows about 42% of its
  width; the card lined up inside the guides arrived at about a fifth of the frame area, with
  scene the user could not see around it. `arTick` now posts exactly the visible rectangle
  (`coverCrop()`), long side 540, and `drawARHUD` maps the returned quad back through that crop.
  On a 1920x1080 stream in a 412x915 viewport the card is now 2.4x larger in the posted frame.
- **AR guidance repeated a canned line.** When detection refuses, the session now passes the
  detector's own reason through ("could not locate a card-shaped quadrilateral. Shoot the card
  against a plain contrasting background with all four edges visible") instead of
  "point at a card, all four edges in frame".

### Added
- `tests/browser/test_ar_loop.py`: the live-AR loop in Chromium with a fake camera, against a
  stub server. Pins that frames are posted, that the posted frame is the cover-cropped region,
  that the quad maps back through the crop, that the inset is painted, and that a 502 HTML
  reply surfaces as an error. Skipped where playwright or chromium is not installed.

## [2.14.0] - 2026-09-16

### Added
- **Private trainer (`trainer/`, `Dockerfile.trainer`, compose profile `trainer`).** Nightly
  loop against your curated card photos: `trainer-sync` mirrors the Google Drive folder with
  rclone (`drive.readonly` scope enforced, `copy` never deletes), and the `trainer` service
  (`network_mode: none`) runs `day` (manifest diff keyed by SHA-256, one-time PDF
  rasterisation, split, pseudo-labels, review queue), `night` (fit on confirmed train labels,
  evaluate against the champion on the frozen test set) and `dawn` (one promotion decision:
  paired-bootstrap gain outside the error bars and no regression in any resolution slice).
  Splits group by physical card (`card_uid`) and a 256-bit near-duplicate hash; frozen test
  items never move, and a training item later linked to one is quarantined. Pseudo-labels live
  in their own table and never count as confirmed. Tasks: `quad_detect`, `back_hue`.
  `trainer/Run-Trainer.ps1` runs a phase for Task Scheduler. See `docs/PRIVATE_TRAINER.md`.
- **Trainer tables in the local vault.** The manifest, labels, prices and promotion log are the
  `trainer` schema of `supabase_vault.duckdb` (`BAKUGO_VAULT_DB`). A vault inside the repository
  is accepted only while git and `.dockerignore` exclude it and its `.wal`
  (`cardcenter/vault.py`). `cardcenter --transfer-supabase` now also honours `BAKUGO_VAULT_DB`.
- **Release guard (`cardcenter/release_guard.py`).** The public `Dockerfile` refuses a build
  whose package holds a collection-derived or untagged artifact (card index, priced priors,
  sticker dataset, number priors), a nested payload, a hash mismatch or missing lineage.
  `python -m trainer release <task>` publishes a promoted generic artifact to
  `cardcenter/data/released/`.
- **QUIPU proposals, never realisations.** The trainer writes collector-number priors (confirmed
  train labels only) to `<private>/outbox/quipu/` with `realised: false`, and refuses to run
  while it can see `QUIPU_ATTEST_KEY_FILE`, `QUIPU_REALISE_GRANT_REF` or a readable
  `attest.key`.
- **Information floor along the card outline (`cardcenter/edge_information.py`).** The
  Cramer-Rao and shot-noise bounds from `information.py` are now measured on the camera frame
  along each side of the detected quad (per colour channel, outermost resolvable edge, local
  flat bands): per-side contrast, noise, blur, edge offset, corner sigma, the ratio floor, and
  capture advice ranked by how much each change lowers the floor (closer ~ ppm^-1.5, contrast
  ~ 1/C, blur ~ sqrt(sigma_p), light). `locate_card` runs `find_card_quad` and then moves each
  side onto the edge it actually found, kept only if re-measurement confirms it. On synthetic
  captures at the 540 px tracking size, `find_card_quad` placed the outline about one border
  width outside the card in 9 of 10 frames (35-40 px at full resolution); after the snap the
  error was under 3 px.
- **Card-back hue references (`cardcenter/hue_reference.py`).** Hues are no longer hard-coded:
  the registry resolves `BAKUGO_HUE_REFERENCES`, then the released `back_hue` artifact, then
  defaults marked `uncalibrated` where they were never measured. Classification uses the
  distance in sigmas (reference spread and the crop's own standard error from its measured
  noise and chroma), abstains between references, and takes price labels from a
  `LabelProfile`.
- **General price attribution (`cardcenter/pricing.py`).** Any source (sticker, tag,
  shelf_sign, page_label, receipt, listing, verbal, manual), any scope (item, page, box, lot,
  venue) and kind (asking, paid, sold, estimate). `parse_price_text` reads `$2.50`, `2,50 €`,
  `50c`, `3 for $1`, and refuses bare numbers. `resolve_item_price` picks the most specific
  attribution and splits lot prices per item. The trainer stores attributions
  (`confirm-file` `price`, `import-prices`) and exports each test item's resolved price.
- **In-situ learning (`cardcenter/insitu.py`, `docs/IN_SITU_LEARNING.md`).** `/identify` records
  each result per device and returns `identification_id`, the outline, its information floor and
  a decision. `POST /feedback` takes confirm / correct / reject plus a price; it updates that
  device's own encounter counts, which break OCR ties between species names only on clear
  history (>= 3 confirmations, >= 3:1). Requests carrying `CARDCENTER_OWNER_TOKEN`
  (`X-Bakugo-Owner`) also write the confirmed photo, label and price to the private inbox, which
  the trainer imports as confirmed (`owner-device:<id>`). Other devices' feedback reaches the
  trainer only as aggregate review items, without images or device ids.
- **Service-worker second vote (`/sw.js`, `cardcenter/embed.py`).** A module service worker runs
  an ONNX embedding model with a self-hosted onnxruntime-web WASM bundle
  (`CARDCENTER_EMBED_DIR`) and compares the rectified crop with the cards this device has
  confirmed (IndexedDB, never uploaded). `POST /identify/vote` applies
  `confidence.gate_identification`: accept only when OCR and the embedding name the same card,
  the match is strong, the device index has at least 3 cards, every side of the outline is
  resolvable, and the frame is within 1.6x the photon floor. Without a model the worker answers
  `available: false`. Verified in Chromium with onnxruntime-web 1.30.0.
- The web app shows the decision and both votes, and has confirm / fix / wrong buttons with a
  price field and source; Settings has an Owner Token field.
- `/ar/push` returns `information`, `decision` and `decision_reason`.

### Changed
- **AR tracking uses measured uncertainty.** `ARSession` acquires with `locate_card`, snaps the
  tracked outline to its measured edges, sets the 1-euro filter's cutoff and the next normal
  search radius from the achieved corner sigma (line-fit residuals, bounded below by the
  Cramer-Rao floor), refuses to measure when a side has no resolvable edge or the floor
  exceeds the grade gate's limit (with the change that would help most as guidance), and runs
  `confidence.gate` on the fused ratio. `track_quad(..., return_stats=True)` returns the fit
  residuals.
- `ingest.binder_sticker.classify_back_franchise` delegates to `hue_reference`. The orange
  price-bar band is excluded only while it covers less than 35% of the usable pixels, so a
  brown Yu-Gi-Oh back brighter than V = 80 is no longer masked out entirely.
- Package data includes `static/*.js` and `data/released/*/*.json`.
- `.gitignore` / `.dockerignore` exclude the private tier's files, `*.duckdb.wal` and
  credentials; local databases stay on disk and are only kept out of the image build upload.

### Fixed
- **`recognition_eval.py --quipu off` never turned QUIPU off.** `quipu_client.base_url()`
  falls back to `http://127.0.0.1:7100`, so `enabled()` was always true and removing
  `QUIPU_URL` changed nothing. `CARDCENTER_QUIPU_DISABLE=1` now disables the client, and the
  evaluator's off pass sets it.
- `cardcenter.__version__` said 2.11.0 while the package was 2.13.2; both are now 2.14.0.

### Security
- **Tenant identity is issued by the server.** `/my-scans`, `/my-analytics`, `/measure` and the
  AR routes no longer trust `X-Device-ID`, `X-Client-ID`, `?device_id=` or a `device_id` form
  field, which let any caller read another device's scan history. The server now issues a
  random token (`bakugo_device` HttpOnly cookie; `X-Device-Token` header plus
  `Authorization: Bearer` for allowed cross-origin clients such as the mobile shell) and stores
  scans under `dev_<sha256(token)>`. Scans saved under the old browser-generated ids are not
  visible to the new identities. `CARDCENTER_TRUST_DEVICE_HEADER=1` restores the old behaviour
  for a private LAN-only server.
- **`/marketplace/tokenize` and `/marketplace/assets` are off** unless
  `CARDCENTER_ENABLE_MARKETPLACE=1`. The hardcoded default wallet is removed; `wallet` is required.
- **`/quipu` is off** unless `CARDCENTER_EXPOSE_QUIPU=1`.
- **Request limits.** Bodies over `CARDCENTER_MAX_BODY_BYTES` (12 MB) get 413 before any of the
  body is read; chunked bodies get 411; bad `Content-Length` gets 400. Images are checked from
  the JPEG/PNG/WebP header against `CARDCENTER_MAX_IMAGE_PIXELS` (40 MP) before decoding, and
  other formats are refused. Sockets time out after `CARDCENTER_SOCKET_TIMEOUT` (30 s),
  concurrent connections are capped at `CARDCENTER_MAX_CONNECTIONS` (32), and in-memory AR
  sessions at `CARDCENTER_MAX_AR_SESSIONS` (256, LRU). `OPENCV_IO_MAX_IMAGE_PIXELS` is set.
- **CORS is an allowlist** (`CARDCENTER_CORS_ORIGINS`; default `https://bakugo.loadopoly.com`,
  `https://loadopoly.com`, and the Capacitor origins `https://localhost`, `capacitor://localhost`)
  instead of `*`.
- **Errors are generic.** Unhandled exceptions return HTTP 500 with `"internal error"` and a
  `ref`; the traceback goes to the server log only. Responses also carry `nosniff`,
  `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`.
- **Container.** Runs as UID 10001 with a read-only root filesystem, `/tmp` tmpfs, all
  capabilities dropped, `no-new-privileges`, and memory/CPU/PID limits. The DuckDB `sqlite`
  extension is installed at build time because the container has no internet access.
  `docker-compose.yml` puts the app on an `internal` network behind Caddy
  (`deploy/Caddyfile`), publishes only `127.0.0.1:8765`, and adds a one-shot `volume-init`
  service that chowns an existing root-owned `/data` volume. An optional `cloudflared`
  service is available under the `tunnel` profile.

## [2.13.2] - 2026-09-15

### Fixed
- **Live AR box drawn in the wrong place**: `#ar-video` uses `object-fit: cover` (cropped to
  fill the portrait viewport) but `#ar-canvas` was stretched to 100%x100%, so quad corners
  were squashed toward the centre on one axis relative to the video under them. The canvas
  now uses the same `object-fit: cover`, so both crop identically.

### Added
- **Detector debug inset** in Live AR: shows the 540 px frame actually sent to `/ar/push`
  with the raw returned quad in red. Separates detector errors from display-mapping errors.
  Tap to hide.

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
