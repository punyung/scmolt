"""Validate package sources, immutable configuration, and runtime versions."""

from __future__ import annotations

import argparse
import json

from .config import (
    BRIDGE_WEIGHTS,
    EPOCHS,
    METHOD_NAME,
    bridge_config,
    collect_core_source_hashes,
    core_config,
    installed_versions,
    version_drift,
)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-runtime-drift", action="store_true")
    args = parser.parse_args()
    drift = version_drift()
    if drift and not args.allow_runtime_drift:
        raise RuntimeError(f"Runtime version drift: {drift}")
    config = core_config()
    if config.epochs != EPOCHS or config.cycle_scale(29) != 0.0:
        raise RuntimeError("Frozen 65-epoch warmup contract drifted")
    if config.cycle_scale(30) != 1.0 / 30.0 or config.cycle_scale(59) != 1.0:
        raise RuntimeError("Frozen 30-epoch ramp contract drifted")
    if bridge_config().to_dict() != BRIDGE_WEIGHTS:
        raise RuntimeError("Frozen bridge weights drifted")
    receipt = {
        "status": "passed",
        "method_name": METHOD_NAME,
        "core_config": config.to_dict(),
        "bridge_config": bridge_config().to_dict(),
        "source_sha256": collect_core_source_hashes(),
        "package_versions": installed_versions(),
        "runtime_version_drift": drift,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
