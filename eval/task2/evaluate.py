import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from topbrain25_eval.metrics.cls_avg_dice import dice_coefficient_single_label
from topbrain25_eval.metrics.cls_avg_hd95 import hd95_single_label
from topbrain25_eval.utils.utils_mask import extract_labels

N_CLASSES = 52


def load_gt(fn: Path, execute_in_docker: bool = True) -> sitk.Image:
    """
    Load a ground truth reference mask for a given filepath.

    Supports both .mha and .nii.gz files.

    When running in Docker, the input filename is resolved relative to
    /opt/ml/input/data/ground_truth/location_masks after removing
    `_0000` from the filenames.

    Returns a SimpleITK Image.
    """
    print(f"fn = {fn}")

    if execute_in_docker:
        gt_dir = Path("/opt/ml/input/data/ground_truth/location_masks")
        name = fn.name.replace("_0000", "")

        base = name.removesuffix(".nii.gz").removesuffix(".mha")
        matches = [
            *gt_dir.glob(f"{base}.mha"),
            *gt_dir.glob(f"{base}.nii.gz"),
        ]

        if not matches:
            raise FileNotFoundError(f"No ground truth found for {fn}")

        gt_path = matches[0]
    else:
        gt_path = fn

    print(f"load_gt path = {gt_path}")

    return sitk.ReadImage(str(gt_path))


# NOTE: dice, iou, hd95 imported from topbrain25 evaluation.
# volumetric similarity (vs) uses the same boilerplate as dice/iou
def vs_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """
    SimpleITK's GetVolumeSimilarity() computes the volume difference
        VS_sitk = 2(V1-V2)/(V1+V2)
    Convert to conventional VS with this formula:
        VS_conventional = 1 - |VS_sitk|/2
    Ref:
        Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
        BMC medical imaging. 2015 Aug 12;15(1):29.
    """
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0

    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    # SimpleITK's signed output: ranges from -2 to +2
    sitk_vs = overlap_measures.GetVolumeSimilarity(label)

    # Convert to conventional Volumetric Similarity: ranges from 0 to 1
    conventional_vs = 1.0 - (abs(sitk_vs) / 2.0)
    return conventional_vs


def evaluation_function(pred: sitk.Image, gt_path, execute_in_docker=True):
    gt_path = Path(gt_path)

    gt = load_gt(gt_path, execute_in_docker)

    if gt.GetSize() != pred.GetSize():
        raise ValueError(
            f"GT and prediction have different sizes: "
            f"{gt.GetSize()} vs {pred.GetSize()}"
        )

    result = {
        "gt_filename": gt_path.name,
    }

    gt_locs = extract_labels(sitk.GetArrayFromImage(gt))
    pred_locs = extract_labels(sitk.GetArrayFromImage(pred))

    for cls in range(1, N_CLASSES + 1):
        # Detection based on segmentation

        # same as eval/task1/evaluate.py's evaluation_function
        result[f"TP_{cls}"] = int(cls in pred_locs and cls in gt_locs)
        result[f"TN_{cls}"] = int(cls not in pred_locs and cls not in gt_locs)
        result[f"FP_{cls}"] = int(cls in pred_locs and cls not in gt_locs)
        result[f"FN_{cls}"] = int(cls not in pred_locs and cls in gt_locs)

        # also report support
        result[f"support_{cls}"] = int(cls in gt_locs)

        # Segmentation metrics are undefined for true negatives.
        if result[f"TN_{cls}"]:
            result[f"DICE_{cls}"] = np.nan
            result[f"HD95_{cls}"] = np.nan
            result[f"VOLSIM_{cls}"] = np.nan
        else:
            result[f"DICE_{cls}"] = dice_coefficient_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )
            result[f"HD95_{cls}"] = hd95_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )[0]  # [hd95_score, hd100_score]
            result[f"VOLSIM_{cls}"] = vs_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )

    return result


def evaluation_aggregation(results: list):
    """Aggregate evaluation results by class across images.

    Detection counts (TP, TN, FP, FN) are summed across images.
    Then, precision, recall, and MCC are computed from the aggregated
    counts for each class (division-by-zero returns NaN).

    Segmentation metrics (DICE, HD95, VOLSIM) are averaged across
    images, ignoring NaN values.
    """
    aggregates = {}
    keys = results[0].keys()

    # Aggregate based on detection counts vs segmentation metrics
    for k in keys:
        if k == "gt_filename":
            continue

        # all values for this class across images
        values = [result[k] for result in results]

        if k.startswith(("TP_", "TN_", "FP_", "FN_", "support_")):
            # Pool detection counts across images.
            aggregates[k] = sum(values)

        elif k.startswith(("DICE_", "HD95_", "VOLSIM_")):
            # Average over valid seg-metric values for each class.
            valid_values = [v for v in values if not np.isnan(v)]
            aggregates[k] = np.mean(valid_values) if valid_values else np.nan

        else:
            # Unknown metric without a defined aggregation policy
            raise ValueError(f"Undefined aggregation for{k}")

    # Obtain detection metrics from aggregated counts
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

    # aggregates with detection and segmentation metrics for each class
    return aggregates


def nanmean(aggregates, metric_name):
    """return 0 for a metric with all nan for GC leaderboard"""
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
    """
    The average reported here is across classes:
        Across images -> evaluation_aggregation()
        Across classes -> evaluation_average()
    e.g.
        DICE_i = average Dice for class i across valid images.
        DICE = average of DICE_i across the 52 classes.

    If the nanmean of a metric after averaging is still nan,
    for GC leaderboard display, convert the averaged nanmean to 0
    """

    return {
        "PRECISION": nanmean(aggregates, "PRECISION"),
        "RECALL": nanmean(aggregates, "RECALL"),
        "MCC": nanmean(aggregates, "MCC"),
        "DICE": nanmean(aggregates, "DICE"),
        "HD95": nanmean(aggregates, "HD95"),
        "VOLSIM": nanmean(aggregates, "VOLSIM"),
    }
