"""
Run pytest from the script's parent folder:
python3 -m pytest test_evaluations/
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from evaluate import (
    evaluation_aggregation,
    evaluation_average,
    evaluation_function,
    load_gt,
    nanmean,
)

base_dir = Path(__file__).parent


@pytest.fixture
def three_results():
    """
    three predictions to compare with file1_cls_1-42-52_gt.json
    result1 is all correct predictions
    result2 is 1 fp and 1 fn
    result3 is all empty prediction
    """
    gt_path = base_dir / "ground-truth/file1_cls_1-42-52_gt.json"
    return [
        # all correct predictions
        evaluation_function(
            pred_locs=[1, 42, 52], gt_path=gt_path, execute_in_docker=False
        ),
        # 1fp and 1fn predictions
        evaluation_function(
            pred_locs=[7, 42, 52], gt_path=gt_path, execute_in_docker=False
        ),
        # empty predictions
        evaluation_function(pred_locs=[], gt_path=gt_path, execute_in_docker=False),
    ]


def test_load_gt():
    fn = base_dir / "ground-truth/file1_cls_1-42-52_gt.json"
    gt_locs = load_gt(fn, execute_in_docker=False)

    assert gt_locs == [1, 42, 52]

    fn = base_dir / "ground-truth/file4_cls_empty_gt.json"
    gt_locs = load_gt(fn, execute_in_docker=False)

    assert gt_locs == []


##################################################################
#### tests for evaluation_function()
##################################################################


def test_evaluation_function_bothGTPredEmpty():
    """both GT Pred empty list"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=base_dir) as f:
        f.write("[]")
        f_name = f.name
        f.flush()
        result = evaluation_function(
            pred_locs=[], gt_path=Path(f_name), execute_in_docker=False
        )
        # Verify string fields directly
        assert result["gt_filename"] == Path(f_name).name

        for key, value in result.items():
            if key == "gt_filename":
                continue

            # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
            metric, _ = key.rsplit("_", 1)

            # only TN is positive
            if metric == "TN":
                assert value == 1
            else:
                assert value == 0


def test_evaluation_function_readme_example():
    """readme ABCD example
    possible classes={A, B, C, D}, a case with *gt*=[A, B] and *predictions*=[A, C]
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=base_dir) as f:
        f.write("[1, 2]")
        f_name = f.name
        f.flush()
        result = evaluation_function(
            pred_locs=[1, 3], gt_path=Path(f_name), execute_in_docker=False
        )
        # Verify string fields directly
        assert result["gt_filename"] == Path(f_name).name
        unexpected_values = {}

        for key, value in result.items():
            if key == "gt_filename":
                continue

            # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
            metric, cls_str = key.rsplit("_", 1)
            cls_id = int(cls_str)

            # Determine expected value
            if cls_id == 1:  # A
                expected = 1 if metric in {"TP", "support"} else 0
            elif cls_id == 2:  # B
                expected = 1 if metric in {"FN", "support"} else 0
            elif cls_id == 3:  # C
                expected = 1 if metric == "FP" else 0
            else:  # D
                expected = 1 if metric == "TN" else 0

            # Collect any mismatches
            if value != expected:
                unexpected_values[key] = f"got {value}, expected {expected}"

        assert unexpected_values == {}


def test_evaluation_function_1234_example():
    """another simple 3cls case
    possible classes={1, 2, 3, 4}, a case with *gt*=[1, 2, 3] and *predictions*=[1, 2, 4]
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=base_dir) as f:
        f.write("[1,2,3]")
        f_name = f.name
        f.flush()
        result = evaluation_function(
            pred_locs=[1, 2, 4], gt_path=Path(f_name), execute_in_docker=False
        )
        # Verify string fields directly
        assert result["gt_filename"] == Path(f_name).name
        unexpected_values = {}

        for key, value in result.items():
            if key == "gt_filename":
                continue

            # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
            metric, cls_str = key.rsplit("_", 1)
            cls_id = int(cls_str)

            # Determine expected value
            if cls_id in {1, 2}:
                expected = 1 if metric in {"TP", "support"} else 0
            elif cls_id == 3:
                expected = 1 if metric in {"FN", "support"} else 0
            elif cls_id == 4:
                expected = 1 if metric == "FP" else 0
            else:
                expected = 1 if metric == "TN" else 0

            # Collect any mismatches
            if value != expected:
                unexpected_values[key] = f"got {value}, expected {expected}"

        assert unexpected_values == {}


def test_evaluation_function_all_correct_pred(three_results):
    result = three_results[0]

    # Verify string fields directly
    assert result["gt_filename"] == "file1_cls_1-42-52_gt.json"

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {1, 42, 52}:
            expected = 1 if metric in {"TP", "support"} else 0
        else:
            expected = 1 if metric == "TN" else 0

        # Collect any mismatches
        if value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_evaluation_function_1fp1fn_pred(three_results):
    result = three_results[1]

    # Verify string fields directly
    assert result["gt_filename"] == "file1_cls_1-42-52_gt.json"

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {42, 52}:
            expected = 1 if metric in {"TP", "support"} else 0
        elif cls_id == 1:
            expected = 1 if metric in {"FN", "support"} else 0
        elif cls_id == 7:
            expected = 1 if metric == "FP" else 0
        else:
            expected = 1 if metric == "TN" else 0

        # Collect any mismatches
        if value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_evaluation_function_empty_pred(three_results):
    result = three_results[2]

    # Verify string fields directly
    assert result["gt_filename"] == "file1_cls_1-42-52_gt.json"

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {1, 42, 52}:
            expected = 1 if metric in {"FN", "support"} else 0
        else:
            expected = 1 if metric == "TN" else 0

        # Collect any mismatches
        if value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


##################################################################
#### tests for evaluation_aggregation()
##################################################################


def test_evaluation_aggregation_all_correct_pred(three_results):
    """aggregate singular result from test_evaluation_function_all_correct_pred()"""
    # all correct predictions
    result = three_results[0]
    aggregates = evaluation_aggregation([result])

    unexpected_values = {}

    for key, value in aggregates.items():
        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {1, 42, 52}:
            if metric in {"TP", "support", "PRECISION", "RECALL", "F1"}:
                expected = 1
            elif metric == "MCC":
                expected = np.nan
            else:
                expected = 0
        else:
            if metric in {"MCC", "PRECISION", "RECALL", "F1"}:
                expected = np.nan
            elif metric == "TN":
                expected = 1
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_evaluation_aggregation_3results(three_results):
    """aggregate 3 results from
    test_evaluation_function_all_correct_pred()
    test_evaluation_function_1fp1fn_pred()
    test_evaluation_function_empty_pred()

    will aggregate:
    TP_1 = 1
    TP_42 = 1
    TP_52 = 1

    FN_1 = 1
    FP_7 = 1
    TP_42 = 1
    TP_52 = 1

    FN_1 = 1
    FN_42 = 1
    FN_52 = 1
    """
    aggregates = evaluation_aggregation(three_results)

    unexpected_values = {}

    for key, value in aggregates.items():
        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id == 1:
            if metric == "TP":
                expected = 1
            elif metric == "FN":
                expected = 2
            elif metric == "support":
                expected = 3
            elif metric == "RECALL":
                expected = 1 / (1 + 2)
            elif metric == "PRECISION":
                expected = 1 / (1 + 0)
            elif metric == "F1":
                expected = 2 * 1 / (2 * 1 + 0 + 2)  # 0.5
            elif metric == "MCC":
                expected = np.nan
            else:
                expected = 0
        elif cls_id == 7:
            if metric == "FP":
                expected = 1
            elif metric == "TN":
                expected = 2
            elif metric in {"RECALL", "MCC"}:
                expected = np.nan
            else:
                expected = 0
        elif cls_id in {42, 52}:
            if metric == "TP":
                expected = 2
            elif metric == "FN":
                expected = 1
            elif metric == "support":
                expected = 3
            elif metric == "RECALL":
                expected = 2 / (2 + 1)
            elif metric == "PRECISION":
                expected = 2 / (2 + 0)
            elif metric == "F1":
                expected = 2 * 2 / (2 * 2 + 0 + 1)  # 0.8
            elif metric == "MCC":
                expected = np.nan
            else:
                expected = 0
        else:
            if metric in {"MCC", "PRECISION", "RECALL", "F1"}:
                expected = np.nan
            elif metric == "TN":
                expected = 3
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


##################################################################
#### tests for evaluation_average()
##################################################################


def test_nanmean():
    aggregates = {}
    for i in range(1, 53):
        aggregates[f"UZH_{i}"] = 100 + i
        aggregates[f"ZHAW_{i}"] = 10 * i
        aggregates[f"ETH_{i}"] = np.nan

    assert nanmean(aggregates, "UZH") == (26.5 + 100, 52)
    assert nanmean(aggregates, "ZHAW") == (26.5 * 10, 52)
    # for GC leaderboard table display, convert final nan to 0
    assert not np.isnan(nanmean(aggregates, "ETH")[0])
    assert nanmean(aggregates, "ETH") == (0, 0)


def test_evaluation_average_all_correct_pred(three_results):
    """average singular result from test_evaluation_function_all_correct_pred()"""
    # all correct predictions
    result = three_results[0]
    aggregates = evaluation_aggregation([result])

    macro = evaluation_average(aggregates)

    assert macro == {
        "MCC": 0,  # due to all NaN
        "count_valid_MCC": 0,
        "PRECISION": 1.0,
        "count_valid_PRECISION": 3,
        "RECALL": 1.0,
        "count_valid_RECALL": 3,
        "F1": 1.0,
        "count_valid_F1": 3,
    }


def test_evaluation_average_3results(three_results):
    """average 3 results from
    test_evaluation_function_all_correct_pred()
    test_evaluation_function_1fp1fn_pred()
    test_evaluation_function_empty_pred()
    """
    aggregates = evaluation_aggregation(three_results)

    macro = evaluation_average(aggregates)

    assert macro == {
        "MCC": 0,  # due to all NaN
        "count_valid_MCC": 0,
        "PRECISION": (1 + 0 + 1 + 1) / 4,
        "count_valid_PRECISION": 4,
        "RECALL": (1 / 3 + 2 / 3 + 2 / 3) / 3,
        "count_valid_RECALL": 3,
        "F1": (0.5 + 0 + 4 / 5 + 4 / 5) / 4,  # 2.1/4 = 0.525
        "count_valid_F1": 4,
    }
