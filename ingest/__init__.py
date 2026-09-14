"""Binder-photo OCR/VLM ingestion.

Turns a folder of sequentially-photographed trading-card binder pages into
per-card identity + price records, per the front/back grid-mirror pairing
scheme described in ``binder_ingest.py``'s module docstring.

This package is deliberately independent of the rest of ``cardcenter``, not
just the live AR serving path: it is an offline batch tool over a folder of
JPEGs, with zero imports from ``cardcenter`` at all. That was a considered
choice, not an oversight -- ``cardcenter``'s own card-quad detector
(``geometry.py``/``multicard.py``) is tuned to find a SINGLE card by contrast
against its surroundings, and empirically does not work on a whole binder
page (see ``binder_grid.py``'s module docstring for the specific failure
mode tested against real photos). This package's ``binder_grid.py`` finds
the PAGE's own outer quad instead and subdivides it, which is a different
enough problem that sharing code would mean bending cardcenter's detector to
a shape it wasn't built for. The OCR half (``binder_sticker.py``) similarly
reimplements cardcenter.ocr's preprocessing recipe rather than importing it,
since that module's ``preprocess_number_crop`` is tuned for a digit-only
collector-number crop, not sticker/label text. Being self-contained also
means this package is safe to run against a photo folder with no app or
services running, and keeps working even if cardcenter's own modules change
shape.
"""
