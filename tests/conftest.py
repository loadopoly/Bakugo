"""Test-wide settings."""

import os

# The server keeps the stills and measured frames it is sent (cardcenter.
# fieldlog). Tests post hundreds of synthetic images; do not keep them unless
# a test asks to.
os.environ.setdefault("CARDCENTER_FIELD_LOG", "0")

# QUIPU writes go to a throwaway outbox and a closed port unless a test sets
# its own observer: no test can post into the live QUIPU.
import tempfile as _tempfile
os.environ.setdefault("QUIPU_OUTBOX", os.path.join(_tempfile.mkdtemp(prefix="bakugo-quipu-"), "outbox.sqlite"))
os.environ.setdefault("CARDCENTER_QUIPU_URL", "http://127.0.0.1:9")
os.environ.pop("QUIPU_EDGE_KEY", None)
