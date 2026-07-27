#!/usr/bin/env python3

"""Audit exact equality between dual JSONL and SQLite algorithm streams."""

import argparse
import importlib.util
import json
import pathlib
import sys


try:
    from algorithm_frame_store import AlgorithmFrameStoreError, compare_dual_storage
except ImportError:  # catkin devel wrapper and source-tree execution
    _storage_module_path = pathlib.Path(__file__).with_name("algorithm_frame_store.py")
    _storage_spec = importlib.util.spec_from_file_location(
        "compare_algorithm_storage_frame_store", _storage_module_path
    )
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    AlgorithmFrameStoreError = _storage_module.AlgorithmFrameStoreError
    compare_dual_storage = _storage_module.compare_dual_storage


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare every decoded value in the declared dual JSONL/SQLite "
            "algorithm storage representations."
        )
    )
    parser.add_argument("run_dir", metavar="RUN_DIR", type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help=(
            "JSON report path (default: "
            "RUN_DIR/analysis/algorithm_storage_equivalence.json)"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = run_dir / "analysis" / "algorithm_storage_equivalence.json"
    else:
        output_path = output_path.expanduser().resolve()
    try:
        report = compare_dual_storage(run_dir)
    except AlgorithmFrameStoreError as exc:
        print("Storage comparison failed: {}".format(exc), file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "Storage equivalence: {} ({})".format(
            "PASS" if report["equivalent"] else "FAIL",
            output_path,
        )
    )
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    sys.exit(main())
