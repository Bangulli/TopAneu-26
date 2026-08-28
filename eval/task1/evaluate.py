import json
import math
from pathlib import Path

import numpy as np

N_CLASSES = 52


def load_gt(fn, execute_in_docker=True) -> list[int]:
    """
    compatible with both json of a naked list or nested json with `locations` key

    Supports either:
      [1, 2, 3]
    or:
      {"locations": [1, 2, 3]}

    Returns the aneu locations list.
    """
    fn = str(fn)

    print(f"fn = {fn}")

    if execute_in_docker:
        dir = Path("/opt/ml/input/data/ground_truth/location_jsons")
        if ".nii.gz" in fn:
            fn = fn.replace("_0000.", ".").replace(".nii.gz", ".json")
        elif ".mha" in fn:
            fn = fn.replace("_0000.", ".").replace(".mha", ".json")
        gt_path = dir / fn
    else:
        gt_path = fn

    print(f"load_gt path = {gt_path}")

    with open(gt_path, "r") as f:
        gt_locs = json.load(f)

    if isinstance(gt_locs, list):
        return gt_locs

    if isinstance(gt_locs, dict) and "locations" in gt_locs:
        return gt_locs["locations"]

    raise ValueError("Expected a list or an object containing a 'locations' key")


def evaluation_function(pred_locs, gt_path, execute_in_docker=True):
    gt_path = Path(gt_path)

    gt_locs = load_gt(gt_path, execute_in_docker)

    # print(f"gt_locs = {gt_locs}")
    # print(f"pred_locs = {pred_locs}")

    result = {}
    result["gt_filename"] = str(gt_path.name)

    for cls in range(1, N_CLASSES + 1):
        result[f"TP_{cls}"] = int(cls in pred_locs and cls in gt_locs)
        result[f"FP_{cls}"] = int(cls in pred_locs and cls not in gt_locs)
        result[f"FN_{cls}"] = int(cls not in pred_locs and cls in gt_locs)
        result[f"TN_{cls}"] = int(cls not in pred_locs and cls not in gt_locs)
        # also report support
        result[f"support_{cls}"] = int(cls in gt_locs)
    return result


def evaluation_aggregation(results: list):
    """for division-by-zero, return NaN"""
    aggregates = {}
    keys = results[0].keys()
    ## aggregate all tpfpfn
    for k in keys:
        if k != "gt_filename":
            aggregates[k] = sum([result[k] for result in results])
    ## compute per location prec, rec, mcc
    for i in range(1, N_CLASSES + 1):
        tp = aggregates[f"TP_{i}"]
        fp = aggregates[f"FP_{i}"]
        fn = aggregates[f"FN_{i}"]
        tn = aggregates[f"TN_{i}"]
        # precision = tp/(tp+fp)
        aggregates[f"PRECISION_{i}"] = tp / (tp + fp) if tp + fp else np.nan
        # recall = tp/(tp+fn)
        aggregates[f"RECALL_{i}"] = tp / (tp + fn) if tp + fn else np.nan
        # mcc = (tp*tn - fp*fn)/sqrt(...)
        mcc_num = tp * tn - fn * fp
        mcc_den = math.sqrt(
            (aggregates[f"TP_{i}"] + aggregates[f"FP_{i}"])
            * (aggregates[f"TP_{i}"] + aggregates[f"FN_{i}"])
            * (aggregates[f"TN_{i}"] + aggregates[f"FP_{i}"])
            * (aggregates[f"TN_{i}"] + aggregates[f"FN_{i}"])
        )
        aggregates[f"MCC_{i}"] = mcc_num / mcc_den if mcc_den else np.nan
    return aggregates


def nanmean(aggregates, metric_name):
    values = np.asarray(
        [aggregates[f"{metric_name}_{i}"] for i in range(1, N_CLASSES + 1)]
    )

    print(f"{metric_name} values = {values}")

    valid_values = [v for v in values if not np.isnan(v)]

    if not valid_values:
        print(f"[WARNING] {metric_name} contains all NaN")
        return 0

    return np.mean(valid_values)


def evaluation_average(aggregates):
    """If the nanmean of a metric is still nan,
    for GC leaderboard display, convert the averaged nanmean to 0"""

    return {
        "PRECISION": nanmean(aggregates, "PRECISION"),
        "RECALL": nanmean(aggregates, "RECALL"),
        "MCC": nanmean(aggregates, "MCC"),
    }
