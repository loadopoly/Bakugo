"""Bakugo private trainer: sync -> manifest -> pseudo-label -> review ->
train -> evaluate -> promote, run against a private data tier that the public
app and the Cloudflare tunnel cannot reach.

The package lives outside ``cardcenter/`` on purpose: the public Dockerfile
copies only ``cardcenter/``, so nothing here ships in the public image.

Invariants this package enforces in code (each has a test):

* The private data root must not sit inside the repository or the Docker
  build context (``paths.resolve_layout``).
* The Drive remote must be configured with the ``drive.readonly`` scope, and
  the mirror is copy-only, so a file removed from Drive never silently removes
  a frozen test item (``sync``).
* Pseudo-labels live in their own table and never become ``confirmed``;
  training reads confirmed labels on the train split only, evaluation reads
  confirmed labels on the frozen test split only (``manifest``, ``data``).
* A test group, once frozen, never moves. A later confirmation that links a
  training item to a frozen test group quarantines the training item
  (``split``).
* Promotion needs a paired improvement outside the error bars and no slice
  regression (``promote``).
* The trainer process refuses to run while holding QUIPU realisation
  credentials, and QUIPU/Supabase sync is forced off (``isolation``).
* Released artifacts carry lineage, and the public build refuses anything that
  encodes collection content (``cardcenter.release_guard``).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
