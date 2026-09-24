"""Test-wide settings."""

import os

# The server keeps the stills and measured frames it is sent (cardcenter.
# fieldlog). Tests post hundreds of synthetic images; do not keep them unless
# a test asks to.
os.environ.setdefault("CARDCENTER_FIELD_LOG", "0")
