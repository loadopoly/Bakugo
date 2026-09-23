"""CLI: python -m trainer <command>

  sync                     rclone copy + listing (network; trainer-sync service)
  day | night | dawn       the loop phases (offline; trainer service)
  status                   counts, champions, recent decisions
  confirm                  confirm one item's identity (human)
  confirm-file FILE        bulk confirmations from a JSONL file (human)
  confirm-quad             confirm one card outline (human)
  import-prices FILE       price attributions from a JSONL file (any source/scope)
  release TASK             copy a generic champion into cardcenter/data/released
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import isolation, loop, manifest, split
from .paths import resolve_layout


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _record_price(con, spec: dict, *, by: str, sha256=None, card_uid=None) -> None:
    """``spec`` is either a full attribution (amount, currency, source, scope,
    scope_key, kind, quantity, venue, ...) or ``{"text": "3 for $1", ...}``."""
    from cardcenter.pricing import PriceAttribution, PriceError, parse_price_text

    fields = dict(spec)
    text = fields.pop("text", None)
    if text is not None:
        parsed = parse_price_text(text)
        if parsed is None:
            raise PriceError(f"no price found in {text!r}")
        fields.setdefault("raw_text", text)
        fields.update({k: v for k, v in parsed.items() if k not in spec})
    for k in ("sha256", "card_uid"):
        fields.pop(k, None)
    att = PriceAttribution.from_dict(fields)
    manifest.attribute_price(con, att, recorded_by=by, sha256=sha256 if att.scope == "item" else None,
                             card_uid=card_uid if att.scope == "item" else None)


def _require_offline() -> bool:
    return os.environ.get("BAKUGO_TRAINER_REQUIRE_OFFLINE", "1") != "0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trainer")
    ap.add_argument("--root", help="private data root (default: $BAKUGO_PRIVATE_ROOT)")
    ap.add_argument("--vault", help="local vault DuckDB (default: $BAKUGO_VAULT_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    d = sub.add_parser("day")
    d.add_argument("--no-predict", action="store_true", help="skip pseudo-labelling")
    d.add_argument("--link-sessions", action="store_true")
    sub.add_parser("night")
    sub.add_parser("dawn")
    sub.add_parser("status")
    c = sub.add_parser("confirm")
    c.add_argument("--sha", required=True)
    c.add_argument("--by", required=True)
    c.add_argument("--label")
    c.add_argument("--number", type=int)
    c.add_argument("--card-uid")
    c.add_argument("--franchise")
    c.add_argument("--source-type")
    c.add_argument("--px-per-mm", type=float)
    c.add_argument("--note", default="")
    for k in ("page-key", "box-key", "lot-key", "venue"):
        c.add_argument(f"--{k}")
    c.add_argument("--price", help='price text, e.g. "$2.50" or "3 for $1"')
    c.add_argument("--price-source", default="manual")
    cf = sub.add_parser("confirm-file")
    cf.add_argument("path")
    cf.add_argument("--by", required=True)
    cq = sub.add_parser("confirm-quad")
    cq.add_argument("--sha", required=True)
    cq.add_argument("--index", type=int, default=0)
    cq.add_argument("--corners", required=True, help='JSON [[x,y],[x,y],[x,y],[x,y]]')
    cq.add_argument("--by", required=True)
    ip = sub.add_parser("import-prices")
    ip.add_argument("path")
    ip.add_argument("--by", required=True)
    r = sub.add_parser("release")
    r.add_argument("task")
    args = ap.parse_args(argv)

    layout = resolve_layout(args.root, vault=args.vault).ensure()

    if args.cmd == "sync":
        from .sync import SyncConfig, run_sync

        isolation.assert_no_realisation_credentials()
        isolation.disable_side_channels()
        result = run_sync(SyncConfig.from_env(), layout)
        loop.log(layout, "sync", **result)
    elif args.cmd in ("day", "night", "dawn"):
        mode = isolation.enter_trainer_mode(require_offline=_require_offline())
        if args.cmd == "day":
            predictors = []
            if not args.no_predict:
                from .pseudo import CardcenterPredictor

                predictors.append(CardcenterPredictor())
            result = loop.day(layout, predictors,
                              split.SplitConfig(link_sessions=args.link_sessions))
        elif args.cmd == "night":
            result = loop.night(layout)
        else:
            result = loop.dawn(layout)
        result = {"isolation": mode, "result": result}
    elif args.cmd == "status":
        result = loop.status(layout)
    elif args.cmd in ("confirm", "confirm-file", "confirm-quad", "import-prices"):
        con = manifest.connect(layout)
        try:
            if args.cmd == "confirm":
                manifest.confirm(con, args.sha, confirmed_by=args.by, label=args.label,
                                 number=args.number, card_uid=args.card_uid,
                                 franchise=args.franchise, source_type=args.source_type,
                                 px_per_mm=args.px_per_mm, note=args.note,
                                 page_key=args.page_key, box_key=args.box_key,
                                 lot_key=args.lot_key, venue=args.venue)
                if args.price:
                    _record_price(con, {"text": args.price, "source": args.price_source,
                                        "venue": args.venue or ""},
                                  by=args.by, sha256=args.sha, card_uid=args.card_uid)
                n = 1
            elif args.cmd == "import-prices":
                n = 0
                for e in _jsonl(args.path):
                    _record_price(con, e, by=args.by, sha256=e.get("sha256"),
                                  card_uid=e.get("card_uid"))
                    n += 1
            elif args.cmd == "confirm-file":
                n = 0
                for e in _jsonl(args.path):
                    c = e.get("confirm")
                    if not c or e.get("task") == "public_feedback":
                        continue    # unedited rows; public feedback is review-only
                    if e.get("task") == "quad":
                        for i, corners in enumerate(c.get("quads", [])):
                            manifest.confirm_quad(con, e["sha256"], i, corners, confirmed_by=args.by)
                    else:
                        manifest.confirm(con, e["sha256"], confirmed_by=args.by,
                                         label=c.get("label"), number=c.get("number"),
                                         card_uid=c.get("card_uid"), franchise=c.get("franchise"),
                                         source_type=c.get("source_type"),
                                         px_per_mm=c.get("px_per_mm"), note=c.get("note", ""),
                                         page_key=c.get("page_key"), box_key=c.get("box_key"),
                                         lot_key=c.get("lot_key"), venue=c.get("venue"))
                        if c.get("price"):
                            _record_price(con, c["price"], by=args.by, sha256=e["sha256"],
                                          card_uid=c.get("card_uid"))
                    n += 1
            else:
                manifest.confirm_quad(con, args.sha, args.index, json.loads(args.corners),
                                      confirmed_by=args.by)
                n = 1
            stats = split.assign(con)      # a confirmation can link components
            result = {"confirmed": n, "splits": stats}
        finally:
            con.close()
    elif args.cmd == "release":
        from .artifacts import release

        result = {"released": str(release(layout, args.task))}
    else:  # pragma: no cover
        ap.error(args.cmd)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
