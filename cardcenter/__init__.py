"""cardcenter -- objective trading card centering measurement.

Measures what can be measured from a photograph: the ratio of opposing border
widths, with a real error bar. Reports a grade *band*, never a grade.

It does not assess corners, edges, or surface, and it does not predict a final
grade. Centering sets a ceiling; the other three attributes decide where under
that ceiling a card actually lands.
"""

__version__ = "2.0.0"

from .centering import measure_centering
from .connection import ConnectionManager, ConnectionSpec, EndpointHealth, SyncClient, SyncPayload, SyncResult
from .grading import (
    CardGradePrediction,
    GradeBand,
    all_grade_bands,
    available_graders,
    grade_band,
    predict_all_grades,
    predict_overall_grade,
)
from .types import (
    SLAB_PRESETS,
    BorderPair,
    CaptureSpec,
    CenteringResult,
    DetectionError,
    DetectionQuality,
    Measured,
    SlabSpec,
)
from .versioning import (
    EngineCapabilities,
    SemVer,
    UpdateCheckResult,
    VersionInfo,
    check_for_updates,
    get_engine_capabilities,
    get_version_info,
)

__all__ = [
    "measure_centering",
    "grade_band",
    "all_grade_bands",
    "available_graders",
    "GradeBand",
    "CardGradePrediction",
    "predict_overall_grade",
    "predict_all_grades",
    "CenteringResult",
    "BorderPair",
    "Measured",
    "SlabSpec",
    "CaptureSpec",
    "SLAB_PRESETS",
    "DetectionError",
    "DetectionQuality",
    "SemVer",
    "VersionInfo",
    "EngineCapabilities",
    "UpdateCheckResult",
    "get_version_info",
    "get_engine_capabilities",
    "check_for_updates",
    "migrate_database",
    "ConnectionManager",
    "ConnectionSpec",
    "EndpointHealth",
    "SyncPayload",
    "SyncResult",
    "__version__",
]

