import numpy as np
import json, os, math, glob
from pathlib import Path
import SimpleITK as sitk

### copied from main
def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = sitk.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return sitk.GetArrayFromImage(result)

def load_gt(fn):
    dir = Path("/opt/ml/input/data/ground_truth/labels")
    return load_image_file_as_array(dir/fn.replace("_0000.nii.gz", ".nii.gz"))

def evaluation_function(predictions, filename): # numpy array of the predictions and the filename 
    gts = load_gt(filename)["locations"]
    
    raise NotImplementedError
    n_aneu = sum(gts.values())
    results = {}
    for cls in range(1, 51): 
        pred_count = preds[cls]
        gt_count = gts[cls]
        tp = min(pred_count, gt_count)
        fp = max(0, pred_count - gt_count)
        fn = max(0, gt_count - pred_count)
        tn = n_aneu-(tp+fn)
        results[f"TP_{cls}"] = tp
        results[f"FP_{cls}"] = fp
        results[f"FN_{cls}"] = fn
        results[f"TN_{cls}"] = tn
    return results

def cls_evaluation_aggregation(metrics, eps=1e-6): # epsilons to avoid div by 0 error
    aggregates = {}
    keys = metrics[0].keys()
    ## aggregate all tpfpfn
    for k in keys:
        aggregates[k]=sum([samp[k] for samp in metrics])
    ## compute per location prec, rec, mcc
    for i in range(1, 51):
        aggregates[f"PRECISION_{i}"] = aggregates[f"TP_{i}"]/(aggregates[f"TP_{i}"]+aggregates[f"FP_{i}"]+eps)
        aggregates[f"RECALL_{i}"] = aggregates[f"TP_{i}"]/(aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+eps)
        mcc_num = aggregates[f"TP_{i}"]*aggregates[f"TN_{i}"] - aggregates[f"FN_{i}"]*aggregates[f"FP_{i}"]
        mcc_den = math.sqrt(
            (aggregates[f"TP_{i}"]+aggregates[f"FP_{i}"])
            *
            (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"])
            *
            (aggregates[f"TN_{i}"]+aggregates[f"FP_{i}"])
            *
            (aggregates[f"TN_{i}"]+aggregates[f"FN_{i}"])
        ) + +eps
        aggregates[f"MCC_{i}"] = mcc_num/mcc_den
    return aggregates

def cls_evaluation_average(metrics):
    averages = {"PRECISION": 0, "RECALL":0, "MCC":0}
    for i in range(1, 51):
        averages["PRECISION"]+=metrics[f"PRECISION_{i}"]
        averages["RECALL"]+=metrics[f"RECALL_{i}"]
        averages["MCC"]+=metrics[f"MCC_{i}"]
    return {k:v/50 for k, v in averages.items()}