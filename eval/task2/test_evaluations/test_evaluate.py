"""
Run pytest from the script's parent folder:
python3 -m pytest test_evaluations/
"""

import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from evaluate import (
    evaluation_aggregation,
    evaluation_average,
    evaluation_function,
    load_gt,
    nanmean,
    vs_single_label,
)

base_dir = Path(__file__).parent


@pytest.fixture
def three_results():
    """
    To test the detection metrics match with task1
    three predictions to compare with file1_cls_1-42-52_gt.json
    result1 is all correct predictions
    result2 is 1 fp and 1 fn
    result3 is all empty prediction
    """
    gt = sitk.GetImageFromArray(np.array([[[0, 1, 42, 52]]], dtype=np.uint8))

    with tempfile.NamedTemporaryFile(suffix=".mha", dir=base_dir) as f:
        tmp_gt_path = Path(f.name)
        sitk.WriteImage(gt, tmp_gt_path)

        yield [
            # all correct predictions
            evaluation_function(
                pred=sitk.GetImageFromArray(
                    np.array([[[0, 1, 42, 52]]], dtype=np.uint8)
                ),
                gt_path=tmp_gt_path,
                execute_in_docker=False,
            ),
            # 1fp and 1fn predictions
            evaluation_function(
                pred=sitk.GetImageFromArray(
                    np.array([[[0, 7, 42, 52]]], dtype=np.uint8)
                ),
                gt_path=tmp_gt_path,
                execute_in_docker=False,
            ),
            # empty predictions
            evaluation_function(
                pred=sitk.GetImageFromArray(np.array([[[0, 0, 0, 0]]], dtype=np.uint8)),
                gt_path=tmp_gt_path,
                execute_in_docker=False,
            ),
        ]


def test_load_gt():
    fn = base_dir / "ground-truth/file1_cls_1-42-52_gt.nii.gz"
    gt = load_gt(fn, execute_in_docker=False)
    assert gt.GetSize() == (5, 6, 4)
    arr = sitk.GetArrayFromImage(gt)
    unique_vals = np.unique(arr).tolist()
    assert unique_vals == [0, 1, 42, 52]

    fn = base_dir / "ground-truth/file2_cls_7_gt.mha"
    gt = load_gt(fn, execute_in_docker=False)
    assert gt.GetSize() == (8, 8, 8)
    arr = sitk.GetArrayFromImage(gt)
    unique_vals = np.unique(arr).tolist()
    assert unique_vals == [0, 7]

    fn = base_dir / "ground-truth/file4_cls_empty_gt.mha"
    gt = load_gt(fn, execute_in_docker=False)
    assert gt.GetSize() == (5, 6, 4)
    arr = sitk.GetArrayFromImage(gt)
    unique_vals = np.unique(arr).tolist()
    assert unique_vals == [0]


##################################################################
#### tests for vs_single_label()
#### other segmentation metrics were tested in topbrain25_eval
##################################################################
def test_vs_single_label():
    # identical_segmentations_return_one
    gt = sitk.GetImageFromArray(np.array([[0, 1, 0]]).astype(np.uint8))
    assert vs_single_label(gt=gt, pred=gt, label=1) == 1

    # missing_label_in_prediction_returns_zero
    gt = sitk.GetImageFromArray(np.array([[0, 1, 0]]).astype(np.uint8))
    pred = sitk.GetImageFromArray(np.array([[0, 0, 0]]).astype(np.uint8))
    assert vs_single_label(gt=gt, pred=pred, label=1) == 0

    # missing_label_in_both_returns_zero
    gt = sitk.GetImageFromArray(np.array([[0, 1, 0]]).astype(np.uint8))
    assert vs_single_label(gt=gt, pred=gt, label=2026) == 0

    # different_volume_sizes: GT has 4 voxels, prediction has 2 voxels
    gt = sitk.GetImageFromArray(
        np.array(
            [
                [6, 6, 0],
                [6, 14, 0],
                [0, 0, 6],
            ]
        ).astype(np.uint8)
    )
    pred = sitk.GetImageFromArray(
        np.array(
            [
                [1, 16, 6],
                [14, 1, 0],
                [6, 0, 0],
            ]
        ).astype(np.uint8)
    )
    assert vs_single_label(gt=gt, pred=pred, label=6) == 1 - 2 / 6


##################################################################
#### tests for evaluation_function()
##################################################################


def test_evaluation_function_bothGTPredEmpty():
    """both GT Pred empty list"""

    blank_img = sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.uint8))

    with tempfile.NamedTemporaryFile(suffix=".mha", dir=base_dir) as f:
        f_name = f.name
        sitk.WriteImage(blank_img, f_name)

        result = evaluation_function(
            pred=blank_img, gt_path=Path(f_name), execute_in_docker=False
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
            # other detections are 0
            elif metric in {"TP", "FP", "FN", "support"}:
                assert value == 0
            # all seg metrics are NaN
            else:
                assert np.isnan(value)


def test_evaluation_function_zeroOverlap():
    """
    GT Pred 0 overlap for all classes, but the location list is the same
    all classes are TP in detection, but with 0 Dice, 1 Volsim, and various HD95
    """
    gt = sitk.GetImageFromArray(np.array([[[0, 1, 2, 3, 4]]], dtype=np.uint8))

    pred = sitk.GetImageFromArray(np.array([[[4, 3, 0, 1, 2]]], dtype=np.uint8))

    with tempfile.NamedTemporaryFile(suffix=".mha", dir=base_dir) as f:
        f_name = f.name
        sitk.WriteImage(gt, f_name)

        result = evaluation_function(
            pred=pred, gt_path=Path(f_name), execute_in_docker=False
        )
        # Verify string fields directly
        assert result["gt_filename"] == Path(f_name).name

        for key, value in result.items():
            if key == "gt_filename":
                continue

            # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
            metric, cls_str = key.rsplit("_", 1)
            cls_id = int(cls_str)

            if cls_id in list(range(1, 5)):
                if metric in {"TP", "support", "VOLSIM"}:
                    assert value == 1
                elif metric == "DICE":
                    assert value == 0
                elif metric == "HD95":
                    if cls_id in {1, 2, 3}:
                        assert value == 2
                    else:  # 4
                        assert value == 4
                else:
                    assert value == 0
            else:
                # remaining classes all TN
                if metric == "TN":
                    assert value == 1
                # other detections are 0
                elif metric in {"TP", "FP", "FN", "support"}:
                    assert value == 0
                # all seg metrics are NaN
                else:
                    assert np.isnan(value)


def test_evaluation_function_readme_example():
    """readme ABCDEF example
    possible classes={ABCDEF}, a case with *gt*=[ABCD] and *predictions*=[ABCE] (B with 0 iou)
    """
    gt = sitk.GetImageFromArray(
        np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 2, 0],
                    [0, 1, 1, 1, 0, 2, 0],
                    [0, 1, 1, 1, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 4, 4, 0, 3, 3, 3],
                    [0, 0, 0, 0, 0, 3, 0],
                ]
            ],
            dtype=np.uint8,
        )
    )

    pred = sitk.GetImageFromArray(
        np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 2],
                    [0, 0, 1, 1, 1, 0, 2],
                    [0, 0, 1, 1, 1, 0, 2],
                    [0, 0, 5, 0, 0, 0, 2],
                    [0, 0, 5, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.uint8,
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".mha", dir=base_dir) as f:
        f_name = f.name
        sitk.WriteImage(gt, f_name)

        result = evaluation_function(
            pred=pred, gt_path=Path(f_name), execute_in_docker=False
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
                if metric in {"TP", "support", "HD95", "VOLSIM"}:
                    expected = 1
                elif metric == "DICE":
                    expected = 12 / 18
                else:
                    expected = 0
            elif cls_id == 2:  # B
                if metric in {"TP", "support"}:
                    expected = 1
                elif metric == "VOLSIM":
                    expected = pytest.approx(6 / 7)
                elif metric == "HD95":
                    expected = pytest.approx(np.percentile([math.sqrt(2), 1, 1, 1], 95))
                else:
                    expected = 0
            elif cls_id == 3:  # C
                if metric in {"TP", "support"}:
                    expected = 1
                elif metric in {"DICE", "VOLSIM"}:
                    expected = 2 / 5
                elif metric == "HD95":
                    expected = pytest.approx(np.percentile([2, math.sqrt(2), 1, 0], 95))
                else:
                    expected = 0
            elif cls_id == 4:  # D
                if metric in {"FN", "support"}:
                    expected = 1
                elif metric == "HD95":
                    expected = 290
                else:
                    expected = 0
            elif cls_id == 5:  # E
                if metric == "FP":
                    expected = 1
                elif metric == "HD95":
                    expected = 290
                else:
                    expected = 0
            else:  # F
                if metric == "TN":
                    expected = 1
                elif metric in {"DICE", "HD95", "VOLSIM"}:
                    expected = np.nan
                else:
                    expected = 0

            # Collect any mismatches
            # mismatch = NOT(both NaN) AND (values differ)
            if not (np.isnan(value) and np.isnan(expected)) and value != expected:
                unexpected_values[key] = f"got {value}, expected {expected}"

        assert unexpected_values == {}


def test_evaluation_function_all_correct_pred(three_results):
    result = three_results[0]

    # Verify string fields directly
    assert re.match(r"^tmp.*\.mha$", result["gt_filename"])

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {1, 42, 52}:
            expected = 1 if metric in {"TP", "support", "DICE", "VOLSIM"} else 0
        else:
            if metric == "TN":
                expected = 1
            elif metric in {"DICE", "HD95", "VOLSIM"}:
                expected = np.nan
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_evaluation_function_1fp1fn_pred(three_results):
    result = three_results[1]

    # Verify string fields directly
    assert re.match(r"^tmp.*\.mha$", result["gt_filename"])

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {42, 52}:
            expected = 1 if metric in {"TP", "support", "DICE", "VOLSIM"} else 0
        elif cls_id == 1:
            if metric in {"FN", "support"}:
                expected = 1
            elif metric == "HD95":
                expected = 290
            else:
                expected = 0
        elif cls_id == 7:
            if metric == "FP":
                expected = 1
            elif metric == "HD95":
                expected = 290
            else:
                expected = 0
        else:
            if metric == "TN":
                expected = 1
            elif metric in {"DICE", "HD95", "VOLSIM"}:
                expected = np.nan
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
            unexpected_values[key] = f"got {value}, expected {expected}"

    assert unexpected_values == {}


def test_evaluation_function_empty_pred(three_results):
    result = three_results[2]

    # Verify string fields directly
    assert re.match(r"^tmp.*\.mha$", result["gt_filename"])

    unexpected_values = {}

    for key, value in result.items():
        if key == "gt_filename":
            continue

        # Parse prefix (e.g. 'TP') and class_id (e.g. 1) from 'TP_1'
        metric, cls_str = key.rsplit("_", 1)
        cls_id = int(cls_str)

        # Determine expected value
        if cls_id in {1, 42, 52}:
            if metric in {"FN", "support"}:
                expected = 1
            elif metric == "HD95":
                expected = 290
            else:
                expected = 0
        else:
            if metric == "TN":
                expected = 1
            elif metric in {"DICE", "HD95", "VOLSIM"}:
                expected = np.nan
            else:
                expected = 0

        # Collect any mismatches
        # mismatch = NOT(both NaN) AND (values differ)
        if not (np.isnan(value) and np.isnan(expected)) and value != expected:
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
            if metric in {"TP", "support", "PRECISION", "RECALL", "DICE", "VOLSIM"}:
                expected = 1
            elif metric == "MCC":
                expected = np.nan
            else:
                expected = 0
        else:
            if metric in {"MCC", "PRECISION", "RECALL", "DICE", "HD95", "VOLSIM"}:
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
            elif metric == "MCC":
                expected = np.nan
            # seg metrics
            elif metric == "DICE":
                expected = (1 + 0 + 0) / 3
            elif metric == "HD95":
                expected = (0 + 290 + 290) / 3
            elif metric == "VOLSIM":
                expected = (1 + 0 + 0) / 3
            else:
                expected = 0
        elif cls_id == 7:
            if metric == "FP":
                expected = 1
            elif metric == "TN":
                expected = 2
            elif metric in {"RECALL", "MCC"}:
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
            elif metric == "MCC":
                expected = np.nan
            # seg metrics
            elif metric == "DICE":
                expected = (1 + 1 + 0) / 3
            elif metric == "HD95":
                expected = (0 + 0 + 290) / 3
            elif metric == "VOLSIM":
                expected = (1 + 1 + 0) / 3
            else:
                expected = 0
        # classes not from {1,7,42,52}
        else:
            if metric in {"MCC", "PRECISION", "RECALL", "DICE", "HD95", "VOLSIM"}:
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

    assert nanmean(aggregates, "UZH") == 26.5 + 100
    assert nanmean(aggregates, "ZHAW") == 26.5 * 10
    # for GC leaderboard table display, convert final nan to 0
    assert not np.isnan(nanmean(aggregates, "ETH"))
    assert nanmean(aggregates, "ETH") == 0


def test_evaluation_average_all_correct_pred(three_results):
    """average singular result from test_evaluation_function_all_correct_pred()"""
    # all correct predictions
    result = three_results[0]
    aggregates = evaluation_aggregation([result])

    macro = evaluation_average(aggregates)

    assert macro == {
        "MCC": 0,
        "PRECISION": 1.0,
        "RECALL": 1.0,
        # 3 classes (1,42,52) have valid seg-metrics
        "DICE": (1 + 1 + 1) / 3,
        "HD95": (0 + 0 + 0) / 3,
        "VOLSIM": (1 + 1 + 1) / 3,
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
        "PRECISION": (1 + 0 + 1 + 1) / 4,
        "RECALL": (1 / 3 + 2 / 3 + 2 / 3) / 3,
        # 4 classes (1,7,42,52) have valid seg-metrics
        "DICE": (1 / 3 + 0 + 2 / 3 + 2 / 3) / 4,
        "HD95": (580 / 3 + 290 + 290 / 3 + 290 / 3) / 4,
        "VOLSIM": (1 / 3 + 0 + 2 / 3 + 2 / 3) / 4,
    }
