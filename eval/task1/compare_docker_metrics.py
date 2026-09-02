#!/usr/bin/env python3
"""
Compare Docker generated metrics.json against expected_metrics.json,
ignoring the gt_filename field and any ordering differences in
the `results` list.

Usage:
    python3 compare_docker_metrics.py <actual_metrics.json> <expected_metrics.json>

Exits 0 if they match, 1 otherwise.
"""

import argparse
import difflib
import json
import re
import sys


def sort_key(record):
    # results may not be in the same order between runs (docker jobs
    # aren't guaranteed to be processed in order), so sort by the
    # numeric file id embedded in gt_filename, e.g. "file3_..." -> 3
    m = re.search(r"file(\d+)", record.get("gt_filename", ""))
    return int(m.group(1)) if m else record.get("gt_filename", "")


def strip_gt_filename(results):
    return [
        {k: v for k, v in record.items() if k != "gt_filename"} for record in results
    ]


def normalize(metrics):
    metrics = dict(metrics)
    metrics["results"] = strip_gt_filename(sorted(metrics["results"], key=sort_key))
    return metrics


def format_json(data):
    """Return deterministic, line-oriented JSON for diffing."""
    return json.dumps(
        data,
        indent=2,
        sort_keys=True,
    ).splitlines(keepends=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", help="path to generated metrics.json")
    parser.add_argument("expected", help="path to expected_metrics.json")
    args = parser.parse_args()

    with open(args.actual) as f:
        actual = json.load(f)

    with open(args.expected) as f:
        expected = json.load(f)

    actual_norm = normalize(actual)
    expected_norm = normalize(expected)

    if actual_norm == expected_norm:
        print(
            "=+= [PASS] metrics.json matches expected_metrics.json (ignoring gt_filename)\n"
        )
        return 0

    print("=+= [MISMATCH] between metrics.json and expected_metrics.json\n")

    diff = difflib.unified_diff(
        format_json(expected_norm),
        format_json(actual_norm),
        fromfile=args.expected,
        tofile=args.actual,
    )

    print("".join(diff))

    return 1


if __name__ == "__main__":
    sys.exit(main())
