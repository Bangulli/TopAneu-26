"""
e2e test for main.py with the 4 test masks in
test_evaluations/ground-truth/
test_evaluations/predictions/

python3 main.py --base_path ./test_evaluations/

The four cases have the following expected segmentation metrics:
    file1_cls_1-42-52_gt:
        class-1:
            Dice = 2*6/(6+8) ~= 0.857
            HD95 = np.percentile([1,1,0,0,0,0,0,0], 95) = 1
            VS = 1 - (8-6)/(8+6) ~= 0.857
        class-42:
            Dice = 0
            HD95 = 290
            VS = 0
        class-52:
            Dice = 2/5
            HD95 = np.percentile([2,1,1,0], 95) = 1.85
            VS = 1 - (4-1)/(4+1) = 2/5
    file2_cls_7_gt:
        class-1:
            Dice = 0
            HD95 = 290
            VS = 0
        class-7:
            Dice = 0
            HD95 = 290
            VS = 0
    file3_cls_1-7-52_gt:
        class-1:
            Dice = 0
            HD95 = 290
            VS = 0
        class-7:
            Dice = 2*3/(6+6) = 0.5
            HD95 = np.percentile([3,sqrt(3),sqrt(2),0,0,0], 95) ~= 2.68
            VS = 1
        class-52:
            Dice = 1
            HD95 = 0
            VS = 1
    file4_cls_empty_gt:
        class-1:
            Dice = 0
            HD95 = 290
            VS = 0
        class-7:
            Dice = 0
            HD95 = 290
            VS = 0
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

test_dir = Path("./test_evaluations/")


def get_result(results, filename):
    return next(r for r in results if r["gt_filename"] == filename)


def check_result(result, *, tp=(), fn=(), fp=(), support=()):
    """
    Assert that a result dict's TP/FN/FP/TN/support values match expectations.

    Pass the class-ids that should be 1 for each metric; anything not
    listed is expected to be 0. `tn` defaults to class-id not
    otherwise mentioned in tp/fn/fp/support
    """
    tp, fn, fp, support = set(tp), set(fn), set(fp), set(support)
    mentioned = tp | fn | fp | support

    for key, value in result.items():
        if not key.startswith(("TP", "FN", "FP", "TN", "support")):
            continue

        print("Check detection result for", key)

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        if metric == "TP":
            expected = int(cls_id in tp)
        elif metric == "FN":
            expected = int(cls_id in fn)
        elif metric == "FP":
            expected = int(cls_id in fp)
        elif metric == "support":
            expected = int(cls_id in support)
        elif metric == "TN":
            expected = int(cls_id not in mentioned)
        else:
            expected = 0

        assert (
            value == expected
        ), f"{result['gt_filename']}: {key} got {value}, expected {expected}"


@pytest.fixture
def saved_metrics():
    """Run main.py e2e, yield the parsed metrics.json, then clean it up."""
    proc_res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--base_path",
            str(test_dir),
        ]
    )
    assert proc_res.returncode == 0

    saved_json_path = test_dir / "output/metrics.json"

    try:
        with saved_json_path.open() as f:
            yield json.load(f)
    finally:
        # clean up
        saved_json_path.unlink(missing_ok=True)


def test_main_e2e_matches_expected_metrics(saved_metrics):
    """the saved metrics.json in test_evaluations/output
    should be identical to the expected_metrics.json"""
    expected_json_path = test_dir / "output/expected_metrics.json"
    with expected_json_path.open() as f:
        expected_metrics = json.load(f)

    assert saved_metrics == expected_metrics


def test_main_e2e_per_file_results(saved_metrics):
    results = saved_metrics["results"]

    # file1 gt_locs = [1, 42, 52], pred_locs = [1, 52]
    result = get_result(results, "file1_cls_1-42-52_gt.nii.gz")
    check_result(result, tp={1, 52}, fn={42}, support={1, 42, 52})
    # seg metrics
    assert result["DICE_1"] == pytest.approx(2 * 6 / (6 + 8))
    assert result["HD95_1"] == 1
    assert result["VOLSIM_1"] == pytest.approx(1 - (8 - 6) / (8 + 6))
    assert result["DICE_42"] == 0
    assert result["HD95_42"] == 290
    assert result["VOLSIM_42"] == 0
    assert result["DICE_52"] == 2 / 5
    assert result["HD95_52"] == pytest.approx(np.percentile([2, 1, 1, 0], 95))
    assert result["VOLSIM_52"] == 2 / 5

    # file2 gt_locs = [7], pred_locs = [1]
    result = get_result(results, "file2_cls_7_gt.mha")
    check_result(result, fn={7}, fp={1}, support={7})
    # seg metrics
    assert result["DICE_1"] == 0
    assert result["HD95_1"] == 290
    assert result["VOLSIM_1"] == 0
    assert result["DICE_7"] == 0
    assert result["HD95_7"] == 290
    assert result["VOLSIM_7"] == 0

    # file3 gt_locs = [1, 7, 52], pred_locs = [7, 52]
    # details for HD95 of class-7:
    #   ref2pred_distances = [1.4142135, 1.0, 1.0, 0, 0, 0]
    #   pred2ref_distances = [3.0, 1.7320508, 1.4142135, 0, 0, 0]
    # -> max(d_95(A,B), d_95(B,A)) -> d_95_pred2ref =  2.683012694120407
    result = get_result(results, "file3_cls_1-7-52_gt.nii.gz")
    check_result(result, tp={7, 52}, fn={1}, support={1, 7, 52})
    # seg metrics
    assert result["DICE_1"] == 0
    assert result["HD95_1"] == 290
    assert result["VOLSIM_1"] == 0
    assert result["DICE_7"] == 0.5
    assert result["HD95_7"] == pytest.approx(
        np.percentile([3, math.sqrt(3), math.sqrt(2), 0, 0, 0], 95)
    )
    assert result["VOLSIM_7"] == 1
    assert result["DICE_52"] == 1
    assert result["HD95_52"] == 0
    assert result["VOLSIM_52"] == 1

    # file4 gt_locs = [], pred_locs = [1, 7]
    result = get_result(results, "file4_cls_empty_gt.mha")
    check_result(result, fp={1, 7})
    assert result["DICE_1"] == 0
    assert result["HD95_1"] == 290
    assert result["VOLSIM_1"] == 0
    assert result["DICE_7"] == 0
    assert result["HD95_7"] == 290
    assert result["VOLSIM_7"] == 0


def test_main_e2e_aggregates(saved_metrics):
    aggregates = saved_metrics["aggregates_per_location"]

    unexpected_values = {}

    for key, value in aggregates.items():
        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id == 1:
            if metric in {"TP", "FN"}:
                expected = 1
            elif metric in {"support", "FP"}:
                expected = 2
            elif metric == "RECALL":
                expected = 1 / (1 + 1)
            elif metric == "PRECISION":
                expected = 1 / (1 + 2)
            elif metric == "F1":
                expected = 2 * 1 / (2 * 1 + 2 + 1)  # 0.4
            elif metric == "MCC":
                expected = (0 - 2) / math.sqrt(3 * 2 * 2 * 1)
            # seg metrics
            elif metric == "DICE":
                expected = pytest.approx(0.857142857 / 4)
            elif metric == "HD95":
                expected = pytest.approx((290 * 3 + 1) / 4)
            elif metric == "VOLSIM":
                expected = pytest.approx(0.857142857 / 4)
            else:
                expected = 0
        elif cls_id == 7:
            if metric in {"TP", "FP", "FN", "TN"}:
                expected = 1
            elif metric == "support":
                expected = 2
            elif metric == "RECALL":
                expected = 1 / (1 + 1)
            elif metric == "PRECISION":
                expected = 1 / (1 + 1)
            elif metric == "F1":
                expected = 2 * 1 / (2 * 1 + 1 + 1)  # 0.5
            elif metric == "MCC":
                expected = (1 - 1) / math.sqrt(2 * 2 * 2 * 2)  # 0
            # seg metrics
            elif metric == "DICE":
                expected = pytest.approx(0.5 / 3)
            elif metric == "HD95":
                expected = pytest.approx((2.68301269 + 290 * 2) / 3)
            elif metric == "VOLSIM":
                expected = pytest.approx(1 / 3)
            else:
                expected = 0
        elif cls_id == 42:
            if metric in {"FN", "support"}:
                expected = 1
            elif metric == "TN":
                expected = 3
            elif metric == "RECALL":
                expected = 0
            elif metric == "PRECISION":
                expected = np.nan
            elif metric == "MCC":
                expected = np.nan
            # seg metrics
            elif metric == "DICE":
                expected = 0
            elif metric == "HD95":
                expected = 290
            elif metric == "VOLSIM":
                expected = 0
            else:
                expected = 0
        elif cls_id == 52:
            if metric in {"TP", "TN", "support"}:
                expected = 2
            elif metric == "RECALL":
                expected = 2 / (2 + 0)
            elif metric == "PRECISION":
                expected = 2 / (2 + 0)
            elif metric == "F1":
                expected = 1
            elif metric == "MCC":
                expected = (4 - 0) / math.sqrt(2 * 2 * 2 * 2)  # 1
            # seg metrics
            elif metric == "DICE":
                expected = pytest.approx((0.4 + 1) / 2)
            elif metric == "HD95":
                expected = pytest.approx(1.85 / 2)
            elif metric == "VOLSIM":
                expected = pytest.approx((0.4 + 1) / 2)
            else:
                expected = 0
        else:
            if metric in {"MCC", "PRECISION", "RECALL", "F1", "DICE", "HD95", "VOLSIM"}:
                expected = np.nan
            elif metric == "TN":
                expected = 4
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_main_e2e_averages(saved_metrics):
    averages = saved_metrics["aggregates_avg"]

    assert averages == pytest.approx(
        {
            # 3 non-nan MCC values
            "MCC": np.mean(
                [
                    -2 / math.sqrt(12),
                    0,
                    1,
                ]
            ),
            "count_valid_MCC": 3,
            # 3 non-nan PRECISION values
            "PRECISION": np.mean(
                [
                    1 / 3,
                    1 / 2,
                    1,
                ]
            ),
            "count_valid_PRECISION": 3,
            # 4 non-nan RECALL values
            "RECALL": np.mean(
                [
                    1 / 2,
                    1 / 2,
                    0,
                    1,
                ]
            ),
            "count_valid_RECALL": 4,
            # 4 non-nan F1 values
            "F1": np.mean(
                [
                    0.4,
                    0.5,
                    0,
                    1,
                ]
            ),
            "count_valid_F1": 4,
            # 4 classes (1,7,42,52) have valid seg-metrics
            "DICE": np.mean(
                [
                    0.857142857 / 4,
                    0.5 / 3,
                    0,
                    (0.4 + 1) / 2,
                ]
            ),
            "count_valid_DICE": 4,
            "HD95": np.mean(
                [
                    (290 * 3 + 1) / 4,
                    (2.68301269 + 290 * 2) / 3,
                    290,
                    1.85 / 2,
                ]
            ),
            "count_valid_HD95": 4,
            "VOLSIM": np.mean(
                [
                    0.857142857 / 4,
                    1 / 3,
                    0,
                    (0.4 + 1) / 2,
                ]
            ),
            "count_valid_VOLSIM": 4,
        }
    )
