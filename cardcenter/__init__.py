"""cardcenter -- objective trading card centering measurement.

Measures what can be measured from a photograph: the ratio of opposing border
widths, with a real error bar. Reports a grade *band*, never a grade.

It does not assess corners, edges, or surface, and it does not predict a final
grade. Centering sets a ceiling; the other three attributes decide where under
that ceiling a card actually lands.
"""

__version__ = "2.22.0"

import os as _os

# OpenCV reads this once, when cv2 is first imported, so it must be set before
# any submodule imports cv2. It caps decoded image size (decompression bombs).
# serve.decode_image also checks the header before decoding.
_os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(64_000_000))

from .centering import measure_centering
from .cloud import CloudConfig, CloudResult, resolve_config, sync_scan_id, sync_store, upsert_scan
from .connection import ConnectionManager, ConnectionSpec, EndpointHealth, SyncClient, SyncPayload, SyncResult
from .world_model_grounding import accumulate_physical_priors, grounding_annotation, physical_world_summary
from .evidence import (
    GOOD_CONSISTENCY,
    MAX_CONSISTENCY,
    Fusion,
    SequentialBoundaryTest,
    Verdict,
    best_single,
    fuse,
    information_value,
)
from .grading import (
    CardGradePrediction,
    GradeBand,
    all_grade_bands,
    available_graders,
    grade_band,
    predict_all_grades,
    predict_overall_grade,
)
from .learning import (
    GradeOutcomeModel,
    LearningStore,
    ingest_certified_labels,
    maybe_load_grade_model,
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
    migrate_database,
)

try:
    from .analytics import AnalyticsEngine, available as analytics_available
    from .transfer_supabase import SupabaseTransferEngine, run_transfer
except ImportError:
    AnalyticsEngine = None  # type: ignore[misc,assignment]
    analytics_available = lambda: False  # noqa: E731
    SupabaseTransferEngine = None  # type: ignore[misc,assignment]
    run_transfer = None  # type: ignore[misc,assignment]

__all__ = [
    "measure_centering",
    "grade_band",
    "all_grade_bands",
    "available_graders",
    "GradeBand",
    "CardGradePrediction",
    "predict_overall_grade",
    "predict_all_grades",
    "GradeOutcomeModel",
    "LearningStore",
    "ingest_certified_labels",
    "maybe_load_grade_model",
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
    "CloudConfig",
    "CloudResult",
    "resolve_config",
    "upsert_scan",
    "sync_scan_id",
    "sync_store",
    "fuse",
    "best_single",
    "SequentialBoundaryTest",
    "Verdict",
    "Fusion",
    "information_value",
    "MAX_CONSISTENCY",
    "GOOD_CONSISTENCY",
    "accumulate_physical_priors",
    "grounding_annotation",
    "physical_world_summary",
    "AnalyticsEngine",
    "analytics_available",
    "SupabaseTransferEngine",
    "run_transfer",
    "__version__",
]
