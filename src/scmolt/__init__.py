"""scMoLT: Gaussian posterior transport for mosaic integration of single-cell multi-omics data."""

from .trackc import TrackCConfig, TrackCMIDASCycle, TrackCVariant

from .anchor_conditioned import build_model as build_anchor_model
from .data import read_predictions, write_midas_input

__version__ = "1.0.0"

__all__ = [
    "TrackCConfig",
    "TrackCMIDASCycle",
    "TrackCVariant",
    "__version__",
    "build_anchor_model",
    "read_predictions",
    "write_midas_input",
]
