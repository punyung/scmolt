"""Public API for the isolated MIDAS-Cycle Joint Embedding Track C."""

from .config import (
    MODALITIES,
    SUPPORTED_TOPOLOGIES,
    TrackCConfig,
    TrackCVariant,
    cycle_weight_at_epoch,
)
from .gaussian import (
    GaussianPosterior,
    diagonal_gaussian_kl,
    gaussian_poe,
    symmetric_kl,
)
from .model import (
    REQUIRED_SCMIDAS_VERSION,
    TrackCBatchDiscriminatorLosses,
    TrackCDiscriminatorLosses,
    TrackCDomainDiscriminatorLosses,
    TrackCForward,
    TrackCLosses,
    TrackCMIDASCycle,
)
from .modules import (
    GaussianResidualTranslator,
    GaussianTransportGraph,
    cycle_consistency_loss,
    identity_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from .replay import GaussianReplayBank
from .remasking import (
    FeatureRemaskedBatch,
    RemaskedModalityBatch,
    remask_observed_features,
    remask_observed_modality,
)

__all__ = [
    "MODALITIES",
    "REQUIRED_SCMIDAS_VERSION",
    "SUPPORTED_TOPOLOGIES",
    "GaussianPosterior",
    "GaussianReplayBank",
    "GaussianResidualTranslator",
    "GaussianTransportGraph",
    "FeatureRemaskedBatch",
    "RemaskedModalityBatch",
    "TrackCConfig",
    "TrackCBatchDiscriminatorLosses",
    "TrackCDiscriminatorLosses",
    "TrackCDomainDiscriminatorLosses",
    "TrackCForward",
    "TrackCLosses",
    "TrackCMIDASCycle",
    "TrackCVariant",
    "cycle_consistency_loss",
    "cycle_weight_at_epoch",
    "diagonal_gaussian_kl",
    "gaussian_poe",
    "identity_loss",
    "lsgan_discriminator_loss",
    "lsgan_generator_loss",
    "remask_observed_features",
    "remask_observed_modality",
    "symmetric_kl",
]
