"""The published DOGMA topology allocation and structural input contract."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from typing import Any, Mapping


POLICY_RESOURCE = "dogma_anchor_conditioned_v1_policy.json"
MODEL_FAMILY = "anchor_conditioned_v1"
REVISION = "20260915_diagonal_full_rna"
CONFIGURATIONS = (
    "full", "paired+full", "paired-abc", "paired-ab", "paired-ac",
    "paired-bc", "diagonal+full", "diagonal",
)
BATCH_ORDER = ("lll_ctrl", "lll_stim", "dig_ctrl", "dig_stim")
MODALITIES = ("rna", "atac", "adt")
_TOPOLOGIES = {
    "full": ("RAT", "RAT", "RAT", "RAT"),
    "paired+full": ("RA", "AT", "RT", "RAT"),
    "paired-abc": ("RA", "RT", "AT", "RT"),
    "paired-ab": ("RA", "AT", "RA", "AT"),
    "paired-ac": ("RA", "RT", "RA", "RT"),
    "paired-bc": ("AT", "RT", "AT", "RT"),
    "diagonal+full": ("A", "R", "T", "RAT"),
    "diagonal": ("A", "R", "T", "T"),
}
_LETTERS = {"R": "rna", "A": "atac", "T": "adt"}
STRUCTURAL_CONFIGS = {
    name: {
        batch: frozenset(_LETTERS[letter] for letter in observed)
        for batch, observed in zip(BATCH_ORDER, rows)
    }
    for name, rows in _TOPOLOGIES.items()
}


def _policy_bytes() -> bytes:
    return files("scmolt.configs").joinpath(POLICY_RESOURCE).read_bytes()


def load_policy() -> dict[str, Any]:
    policy = json.loads(_policy_bytes())
    if (
        policy.get("schema_version") != 1
        or policy.get("model_family") != MODEL_FAMILY
        or policy.get("revision") != REVISION
        or policy.get("epochs") != 65
        or policy.get("seed") != 1234
        or set(policy.get("anchors", {})) != set(CONFIGURATIONS)
        or not set(policy["anchors"].values()).issubset({"RNA", "ATAC", "ADT"})
    ):
        raise ValueError("Packaged anchor-conditioned policy is incompatible")
    return policy


def policy_sha256() -> str:
    return hashlib.sha256(_policy_bytes()).hexdigest()


def selected_anchor(configuration: str) -> str:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown topology {configuration!r}; choose {CONFIGURATIONS}")
    return str(load_policy()["anchors"][configuration]).lower()


def topology_alias(configuration: str) -> str:
    selected_anchor(configuration)
    return configuration if configuration in {"full", "diagonal"} else "paired-abc"


def validate_topology(manifest: Mapping[str, Any], configuration: str) -> None:
    """Reject a topology name that disagrees with the actual observed files.

    Native batch sorting determines training order; BATCH_ORDER only defines
    the observation table and never reorders the input or native modules.
    """
    selected_anchor(configuration)
    observed = manifest["batches"]
    expected = STRUCTURAL_CONFIGS[configuration]
    if set(observed) != set(expected):
        raise ValueError(f"DOGMA input must declare batches {BATCH_ORDER}")
    drift = {
        batch: {
            "expected": sorted(expected[batch]),
            "observed": sorted(observed[batch]["observed_modalities"]),
        }
        for batch in BATCH_ORDER
        if set(observed[batch]["observed_modalities"]) != expected[batch]
    }
    if drift:
        raise ValueError(f"Input topology differs from {configuration!r}: {drift}")


def uses_direct_adt_route(configuration: str, observed: set[str]) -> bool:
    if "adt" in observed or "atac" not in observed:
        return False
    anchor = selected_anchor(configuration)
    return (anchor == "rna" and observed == {"atac"}) or anchor == "atac"
