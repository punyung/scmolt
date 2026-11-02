"""The ``scmolt`` command: model information and input preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _json_output(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _command_info(_args: argparse.Namespace) -> None:
    from .anchor_conditioned.policy import load_policy, policy_sha256

    policy = load_policy()
    _json_output({
        "model_family": policy["model_family"],
        "revision": policy["revision"],
        "anchors": policy["anchors"],
        "policy_sha256": policy_sha256(),
        "epochs": policy["epochs"],
        "seed": policy["seed"],
        "biological_dimensions": 32,
        "technical_dimensions": 2,
        "anchor_fixed_for_entire_run": True,
    })


def _read_layout(path: Path) -> dict[str, list[str]]:
    import pandas as pd

    table = pd.read_csv(path, sep="\t", index_col=0)
    return {str(batch): [m for m in table.columns if int(row[m])] for batch, row in table.iterrows()}


def _command_prepare_input(args: argparse.Namespace) -> None:
    import anndata as ad

    from .data import write_midas_input

    if args.configuration and args.layout:
        raise ValueError("Give --configuration or --layout, not both")
    layout = None
    if args.configuration:
        from .anchor_conditioned.policy import STRUCTURAL_CONFIGS

        layout = {batch: sorted(measured) for batch, measured in STRUCTURAL_CONFIGS[args.configuration].items()}
    elif args.layout:
        layout = _read_layout(args.layout)
    adatas = {m: ad.read_h5ad(getattr(args, m), backed="r") for m in ("rna", "adt", "atac")}
    root = write_midas_input(adatas, args.output_dir, batch_key=args.batch_key,
                             layout=layout, overwrite=args.overwrite)
    print(root)


def build_parser() -> argparse.ArgumentParser:
    from .anchor_conditioned.policy import CONFIGURATIONS

    parser = argparse.ArgumentParser(prog="scmolt", description="scMoLT model information and input preparation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show the anchor of every DOGMA-seq configuration and the recipe")
    info.set_defaults(function=_command_info)

    prepare = subparsers.add_parser(
        "prepare-input", help="write RNA, ADT and ATAC h5ad files as a native MIDAS input directory")
    prepare.add_argument("--rna", type=Path, required=True, help="h5ad with RNA counts")
    prepare.add_argument("--adt", type=Path, required=True, help="h5ad with ADT counts")
    prepare.add_argument("--atac", type=Path, required=True, help="h5ad with ATAC peaks (binarized on writing)")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--batch-key", default="batch", help="obs column with the batch of every cell")
    prepare.add_argument("--configuration", choices=CONFIGURATIONS,
                         help="mask the DOGMA-seq data to one of the eight pseudo-mosaics")
    prepare.add_argument("--layout", type=Path,
                         help="TSV with columns batch, rna, adt, atac (1 = measured) to mask fully measured data")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(function=_command_prepare_input)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"scmolt: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
