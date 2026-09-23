# Measuring at a shop or a stand

What the app does about distance, and where the honest limit is.

## The limit, in numbers

The measurement needs pixels on the card's borders. Below about
**4.5 px/mm** (`capture.MIN_PX_PER_MM`) the border reading stops being
trustworthy, so the app refuses instead of reporting a number. Measured on
synthetic captures at a fixed focal length (`tests/test_framing.py`):

| card width in the photo | px/mm | result |
|---|---|---|
| 677 px | 10.6 | 0.00 pp from truth |
| 508 px | 8.0 | 0.47 pp |
| 313 px | 4.9 | 0.08 pp |
| 254 px | 4.0 | 25 pp out, with an error bar that did not cover the miss — refused |

So: **the card needs to be at least ~300 px wide in the photo you send.**
Everything below is about getting it there without walking closer.

## What the app now does for you

* **Crops to the card before it downsizes.** Both measurement paths used to
  cap the whole frame first (2400 px for a photo, 1200 px live), which threw
  away half the card's resolution to keep a picture of the table. The card is
  located first, and the cap applies to the crop. On a card 313 px wide in a
  4000 px photo that is the difference between a 0.08 pp error and a refusal.
* **Takes the measurement photo at full sensor resolution.** The preview
  stream is 1080p-ish; `takePhoto()` gives the sensor's full frame where the
  browser supports it, which is typically 3-4x more pixels on the card. The
  inset footer says `photo NNNk` when that worked and `frame WxH` when it fell
  back to the preview frame.
* **Sends only what you framed.** The live view is cropped to what you see in
  the viewfinder, not the whole sensor frame.
* **Uploads the card, not the photo.** When the live view is tracking a card,
  the full-resolution still is cropped to that card (plus a wide margin) on
  the phone before it is sent, with its position in the photo so the lens
  geometry is unchanged. Typically a few hundred KB instead of several MB. The
  inset footer says `card crop WxH of WxH`; `full photo` means nothing was
  tracked and the whole photo went.
* **Gives up cleanly on a bad signal.** A still upload that gets no answer in
  60 s says so, and the button comes back. The live loop pauses while a still
  is uploading.
* **Keeps the screen on** while the camera is up.

## Before you leave home

1. The server runs on your PC and the phone reaches it through the Cloudflare
   tunnel, so the PC must be on, awake and running Docker. Check
   `https://bakugo.loadopoly.com/health` from the phone on mobile data (Wi-Fi
   off) — it should report the version you last deployed.
2. Settings → set the owner token if you want the shop photos and prices to
   land in your private inbox for the trainer.
3. Open the app once on mobile data and allow the camera, so the permission
   prompt does not catch you at the counter.

## Data use

The live view sends a ~40-60 KB frame per answer, several a second while
the camera is up: roughly 5-15 MB a minute, depending on how fast answers come back. Close the live view (switch to Photo mode) between cards if
your data is metered. A still is a few hundred KB when it was tracking.

## What to do at the counter

1. **Use the zoom bar under the viewfinder.** If the camera exposes zoom, the
   readout says `2.0x cam` and it is real optical or sensor zoom — this is the
   one that adds detail. If it says `2.0x crop`, the camera gave no zoom
   control and the app is cropping instead: it still helps the detector lock
   on, but it does not add resolution.
2. **Pick the telephoto camera.** Settings → Camera lists every camera the
   phone exposes. A 3x or 5x module reaches further than any crop. Set the
   `Lens` dropdown to match what you chose — it only tells the optics model
   what field of view to assume, it does not switch cameras.
3. **Several cards on the counter: tap the one you want.** The reader takes
   the card under the reticle, or the nearest one if the reticle is on bare
   counter. A tap on a card in the preview picks that card instead (and sets
   the focus there); after that the outline follows the card as you move,
   not the spot you tapped. A thumb over one edge is fine. Two cards lying
   exactly on each other, or two in one holder, get outlined as one: move
   them apart.
4. **All four edges in view, and not too close.** Phone main cameras often
   cannot focus nearer than 10-20 cm; with the card more than half the frame
   wide the live view usually goes soft. Hold the phone a little
   higher and zoom in instead. If the bottom of the card is off screen the
   reader will not take the top half as a card.
5. **Tilt is fine.** Cards lying on the counter at 20-45° are found (the
   reader un-projects the outline and checks it is card-shaped at that
   tilt). Accuracy still drops with tilt, because the far half of the card
   has fewer pixels; for a measurement, the flatter the view the better.
6. **Through glass**: shoot at a slight angle to move the reflection off the
   card, and expect the shot-noise gate to send identifications to review —
   glass plus shop lighting is rarely photon-limited.
7. **In a sleeve or binder page**: fine, as long as one card fills the reticle
   and the page's own edges are not closer to the reticle than the card's.

## If it refuses

The banner says which of the two things is wrong:

* `live view too coarse here (N px/mm) -- hold still and tap Measure Card`
  — the live frame is only 540 px across; the photo has the camera's full
  resolution and the banner says roughly what it will be. This is the normal
  case at a counter.
* `too far even for a photo` / `too far away (N px/mm)` — zoom, switch
  camera, or step in. Nothing else will fix it; the pixels are not there.
* `no card found here` — point the centre of the view at a card or tap one.
* `whole card not in view` — part of the card is off screen; lift the phone
  until all four edges show.
* `outline moved onto the card's edge (top +0.43mm)` — not a refusal: the
  card's edge was soft there (focus, or a sleeve's open end), and the outline
  was put on the edge before the borders were measured. Worth a second shot
  if the number matters.

The outline is drawn from an answer that left the phone up to a second or so
earlier; between answers the page moves it with the preview, so it stays on
the card while you adjust. If it lags anyway (a blank mat gives the page
nothing to match), hold still for a moment.
* `frame is soft -- ... too close to focus` — the card fills most of the view
  and the camera cannot focus that near; lift the phone a little.
* `could not locate a card-shaped quadrilateral` — the outline is not visible:
  a gold border against a dark surface, fingers over a corner, or the card
  flush against other cards. Change what is behind the card, not the distance.
