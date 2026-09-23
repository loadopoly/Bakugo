# Private trainer

The trainer learns from your curated card photos (the cards you buy,
photographed and filed in Google Drive) and from photos your own devices
confirm in the app (see `docs/IN_SITU_LEARNING.md`). It runs
on GARD_DESKTOP, in containers that are not attached to caddy, the tunnel, or
the `backend` network, and only generic parameters that pass a build-time
check can reach the public app.

Code: `trainer/` (not copied into the public image). Guard:
`cardcenter/release_guard.py`. Tests: `tests/test_trainer.py`,
`tests/test_release_guard.py`.

## 1. One-time setup

### Private root and token, outside the repo and outside OneDrive

```powershell
$root = "$env:LOCALAPPDATA\Bakugo\private"
New-Item -ItemType Directory -Force -Path $root | Out-Null
setx BAKUGO_PRIVATE_ROOT_HOST $root
setx BAKUGO_RCLONE_CONFIG_HOST "$env:LOCALAPPDATA\Bakugo\rclone.conf"
setx BAKUGO_DRIVE_FOLDER "Card Captures"      # path inside the remote
```

`trainer/paths.py` refuses a root inside the repository, and
`trainer/Run-Trainer.ps1` also refuses one inside the repository or OneDrive.

### rclone remote with read-only scope

```powershell
$env:RCLONE_CONFIG = "$env:LOCALAPPDATA\Bakugo\rclone.conf"
rclone config create bakugo-drive drive scope drive.readonly
icacls $env:RCLONE_CONFIG /inheritance:r /grant:r "$($env:USERNAME):(R,W)"
```

`python -m trainer sync` reads the config and refuses any scope other than
`drive.readonly` (an unset scope is full access in rclone). If the remote was
first authorised with a wider scope, re-authorise after changing it
(`rclone config reconnect bakugo-drive:`) so the token itself is read-only.
The config file is mounted only into `trainer-sync`.

### Local vault

The trainer's tables (manifest, labels, prices, promotions) are the `trainer`
schema of `supabase_vault.duckdb`, next to the Supabase transfer's `main`
schema. The trainer container mounts the vault's folder, because DuckDB writes
a `.wal` file beside the database:

```powershell
$vault = "$env:LOCALAPPDATA\Bakugo\vault"
New-Item -ItemType Directory -Force -Path $vault | Out-Null
Move-Item supabase_vault.duckdb $vault      # from the repo root, once
setx BAKUGO_VAULT_DIR_HOST $vault
setx BAKUGO_VAULT_DB "$vault\supabase_vault.duckdb"   # cardcenter --transfer-supabase uses it too
```

A vault inside the repository is accepted for runs outside Docker only while
git and `.dockerignore` both exclude it and its `.wal` file
(`trainer/paths.resolve_vault`). DuckDB allows one writing process: do not run
the trainer and `--transfer-supabase` at the same time.

### Build

```powershell
docker compose --profile trainer build
```

## 2. The loop

| Phase | Command | Network | Does |
|---|---|---|---|
| day | `trainer-sync`, then `trainer day` | sync only | rclone copy + Drive IDs; manifest diff over the mirror and the in-situ inbox; PDF rasterisation (once, 200 dpi); owner-device captures applied as confirmed; split assignment; pseudo-labels; review queue (plus aggregate public feedback); QUIPU proposals; recognition and price test-set exports |
| night | `trainer night` | none | fit a candidate per task on confirmed train labels; evaluate it and the champion on the frozen test set |
| dawn | `trainer dawn` | none | one promotion decision per candidate, logged with its reason |

`trainer\Run-Trainer.ps1 -Phase day|night|dawn|status` runs a phase and logs
to `<private>\logs`. Its header has the `schtasks` lines.

## 3. Confirming labels

The day phase writes `<private>\queue\review-YYYY-MM-DD.jsonl`. Each line says
why the item is queued. Every unconfirmed test item is queued whatever the
model's confidence, so the test set is not a sample of the hard cases.

To confirm, add a `confirm` object to the lines you have checked and load the
file:

```json
{"task": "identity", "sha256": "…", "confirm": {"label": "Lechonk", "number": 182, "card_uid": "lechonk-182-a", "franchise": "pokemon", "source_type": "labelled_duplicate", "px_per_mm": 11.4}}
{"task": "quad", "sha256": "…", "confirm": {"quads": [[[x,y],[x,y],[x,y],[x,y]]]}}
```

```powershell
docker compose --profile trainer run --rm trainer confirm-file /private/queue/review-2026-09-16.jsonl --by adam
```

Copy the queue file before editing it; the next day phase rewrites that
day's file.

Prices use any source, not only stickers:

```json
{"task": "identity", "sha256": "…", "confirm": {"label": "Lechonk", "box_key": "bin 4", "venue": "Shop A",
  "price": {"text": "3 for $1", "source": "tag"}}}
```

Prices that cover many items (a bin sign, a page price, a lot) go in a
separate file with `scope` and `scope_key`, and reach items through the
`page_key`/`box_key`/`lot_key`/`venue` you gave them:

```powershell
docker compose --profile trainer run --rm trainer import-prices /private/queue/prices.jsonl --by adam
```

```json
{"amount": "0.25", "source": "shelf_sign", "scope": "box", "scope_key": "bin 4", "kind": "asking"}
```

`card_uid` names the physical card. Give the same value to every capture of
the same card (sleeved, unsleeved, different dates); that is what keeps them
on one side of the split.

## 4. Rules the code enforces

* **Labels.** `files.label_strength` is `weak` or `confirmed` (CHECK
  constraint). Pseudo-labels live in `pseudo_labels` and are never copied.
  `confirm()` needs a human `confirmed_by`. Training reads confirmed train
  items only; evaluation reads confirmed items on the frozen test split only.
* **Split.** Groups are connected components over confirmed `card_uid` and a
  256-bit near-duplicate hash. Test items are frozen. If a later confirmation
  links a training item to a frozen test group, the training item moves to
  `quarantine`; the test item never moves. `--link-sessions` also groups by
  capture session (strict; collapses a binder session into one group).
* **Promotion.** At least 30 paired test items; the paired-bootstrap 95%
  interval of the gain is above zero; no resolution slice drops at all
  (`max_slice_drop = 0`) and none shows a significant drop. Each read of the
  frozen set is counted in `test_uses`.
* **QUIPU.** The trainer writes proposals to `<private>\outbox\quipu\` with
  `realised: false`. It never sends them. It refuses to start if it can see
  `QUIPU_ATTEST_KEY_FILE`, `QUIPU_REALISE_GRANT_REF`,
  `QUIPU_ACCEPT_SELF_ASSERTED=1`, or a readable
  `%LOCALAPPDATA%\QUIPU\attest.key`. Collector-number priors are built from
  train labels only.
* **Isolation.** `trainer` has `network_mode: none` and the day/night/dawn
  commands fail if an outbound connection succeeds. The QUIPU client and the
  Supabase mirror are disabled in the trainer process
  (`CARDCENTER_QUIPU_DISABLE=1`, Supabase variables removed).
* **Release.** `python -m trainer release <task>` (run on the host, with the
  repo checked out) copies a generic champion to
  `cardcenter/data/released/<task>/` with the per-item hash list replaced by
  its digest. The public `Dockerfile` runs `cardcenter.release_guard`, which
  refuses collection kinds (card index, priced priors, sticker dataset,
  number priors), non-generic exposure, nested payloads, hash mismatches,
  missing lineage, and private-tier files.

## 5. Tasks

| Task | Exposure | Trains | Scored by |
|---|---|---|---|
| `quad_detect` | generic | `find_card_quad(min_area_frac)` over a grid, with and without the information snap (`edge_information.locate_card`) | mean corner error ≤ 2% of the diagonal, per labelled card |
| `back_hue` | generic | card-back hue references (mean, spread, count) per franchise with ≥ 3 confirmed backs; replaces the `binder_sticker` constants once released | franchise guess on confirmed test backs |
| number priors | collection | frequencies of confirmed collector numbers | not promoted; filed as a QUIPU proposal |
| recognition | – | nothing (the embedding second vote learns on each device) | `reports/recognition_test.jsonl` for `eval/recognition_eval.py --detect` |
| prices | collection | nothing | `reports/price_test.jsonl`: each frozen test item's resolved price and its source |

## 6. Not done here

* **Quad-detection parameters are not read by the app yet.**
  `release_guard.released_params("quad_detect")` returns a validated payload;
  `find_card_quad` still uses its default `min_area_frac`. The `back_hue`
  release is read (`cardcenter.hue_reference`).
* **Cloudflare Access.** No private HTTP endpoint exists: the review queue is
  a file, and the collection lookup is the vault. If either gets an endpoint,
  put it behind a Cloudflare Access policy on its own hostname, never on
  `bakugo.loadopoly.com`.
