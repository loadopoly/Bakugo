# In-situ learning

What the running app learns from the people using it, what that learning is
allowed to change, and how it reaches the shared models.

Policy: **per-device learning plus owner review**. A user's confirmation
changes that user's own results right away. It changes everyone's results
only after you confirm it in the private trainer.

## 1. The loop

```
phone                         bakugo (public app)                 private tier (GARD_DESKTOP)
-----                         -------------------                 ---------------------------
photo ──► /identify ───────► OCR + outline + information floor
                             record identification (per device)
      ◄── id, name, quad, embed_crop, decision=review
SW: embed crop, kNN over
this device's index
      ──► /identify/vote ───► gate_identification ──► accept | review
user: ✓ / fix / wrong,
price text ──► /feedback ───► feedback row, device prior +1,
                              price attribution (per device)
SW: add embedding to            owner token? ──► inbox/<date>/<id>.jpg + .json ──► trainer day:
this device's index                                                                import as confirmed
                              all other devices ─ aggregate counts, no images ───► review queue
                                                                                   trainer night/dawn:
                                                                                   train, evaluate, promote
                                                                      release ◄─── generic artifacts only
```

## 2. What each confirmation changes

| Change | Who is affected | When |
|---|---|---|
| `device_priors` count for the confirmed name | that device | immediately |
| On-device embedding index (IndexedDB, in the service worker) | that device | immediately |
| Stored price attribution | that device's records | immediately |
| Photo + label + price in the private inbox | nobody until the trainer imports it | owner devices only |
| Shared models (quad detection, back hues, priors) | everyone | after trainer promotion and release |

The device prior is used for one decision: when OCR finds two species names
equally close (`recognise_card` returns alternatives and no name), `/identify`
picks the one this device has confirmed at least 3 times and at least 3:1 more
often (Dirichlet counts, alpha 1). The response says so
(`resolved_by: device_prior`) and the decision stays `review` unless the
second vote agrees.

## 3. The second vote (service worker)

`/sw.js` is a module service worker. When `CARDCENTER_EMBED_DIR` holds a
model it imports a self-hosted onnxruntime-web WASM bundle and:

1. receives the rectified card crop the server returned (`embed_crop`, already
   at the model's input size),
2. runs the ONNX model and L2-normalises the output,
3. compares it with the embeddings this device has confirmed before
   (IndexedDB `bakugo-embed`), best similarity per name,
4. returns `{name, similarity, margin, index_size, model_id}`.

The page posts that to `/identify/vote`. `confidence.gate_identification`
accepts only when all of these hold:

* OCR named the card, and the vote names the same card;
* similarity ≥ 0.80 and margin over the next name ≥ 0.05;
* the device index holds at least 3 cards;
* every side of the outline is resolvable;
* the frame's noise is within 1.6× the photon floor (the same check the grade
  gate uses).

The vote can only confirm or contradict OCR. It never names a card on its own.
Phone JPEGs are often above 1.6× the photon floor, so many agreeing votes will
still come back `review`; the threshold is `IdGateConfig.max_shot_ratio`.

After a confirmation the page asks the worker to `learn` the held embedding
under the confirmed name. Nothing from the index leaves the device.

### Installing a model

```
CARDCENTER_EMBED_DIR/
  model.json
  model.onnx
  ort/ort.wasm.bundle.min.mjs        onnxruntime-web/dist, >= 1.19
  ort/ort-wasm-simd-threaded.wasm    same release
```

`model.json`:

```json
{"model_id": "mobilenetv3-small-1.0", "input_size": [224, 224], "layout": "NCHW",
 "channels": "RGB", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
 "scale": 255, "input_name": "input", "output_name": "embedding",
 "license": "...", "source": "..."}
```

Mount the folder with `BAKUGO_EMBED_DIR_HOST` (read-only). Changing
`model_id` starts a new on-device index; old embeddings are not compared with
new ones. Tested in Chromium with onnxruntime-web 1.30.0. The bundle build is
required: the other builds load their WASM glue with `import()`, which
service workers do not allow. Do not set `ort.env.wasm.wasmPaths`, which also
switches to `import()`.

## 4. Owner devices

Set `CARDCENTER_OWNER_TOKEN` (24+ characters, in `.env`) and
`BAKUGO_INBOX_HOST=<private root>\inbox`. In the app, paste the token under
Settings → Owner Token. Requests with that token keep the uploaded photo in
memory (32 photos, 15 minutes) until you confirm or correct the
identification; then the photo, the label and any price are written to the
inbox. The trainer's day phase imports them with
`confirmed_by = owner-device:<id>` and `source_type = app_capture`.

The token is stored in that browser's localStorage. Use it only on your own
devices.

## 5. Public feedback

Everything other devices report stays in the app's SQLite file. The trainer
reads it read-only (`BAKUGO_APP_DB`) and adds one review item per
(OCR reading, action, user's name) with counts and the number of distinct
devices. No images and no device ids are exported. A review item is never a
label; to use it, confirm the card from your own photos.

## 6. Prices

Stickers are one source among several. `/feedback` accepts
`price: {"text": "3 for $1", "source": "shelf_sign", "scope": "box", "scope_key": "bin 4"}`
or an explicit `amount`/`currency`. Sources: sticker, tag, shelf_sign,
page_label, receipt, listing, verbal, manual. Scopes: item, page, box, lot,
venue. Kinds: asking, paid, sold, estimate. `cardcenter.pricing` resolves an
item's general price from the most specific attribution (item > page > box >
lot > venue; entered before OCR; newest first) and splits lot prices per item.

## 7. Not done

* The per-device character confusion model (`learning.ConfusionModel`) is not
  updated from feedback: `/identify` does not use it yet, and storing counts
  nothing reads would not change a result.
* `quipu_client.observe_measure_async` posts the collector number the system
  read to the QUIPU Observer, and `number_priors()` is built from that
  lexicon. That lets the system's own readings feed its priors, which
  `learning.py`'s circularity rule forbids. The trainer's QUIPU proposals use
  confirmed train labels only; the observer path is unchanged.
