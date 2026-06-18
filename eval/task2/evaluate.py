import numpy as np
import json, os, math, glob
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import label, binary_erosion
from scipy.spatial import cKDTree

### copied from main
def load_image_file_as_array(location):
    result = sitk.ReadImage(location)

    # Convert it to a Numpy array
    return sitk.GetArrayFromImage(result)

def load_gt(fn):
    dir = Path("/opt/ml/input/data/ground_truth/location_masks")
    return load_image_file_as_array(dir/fn.replace("_0000.mha", ".mha"))

def iou(img1, img2):
    return np.bitwise_and(img1, img2).sum()/np.bitwise_or(img1, img2).sum()

def dice(img1, img2):
    return 2*np.bitwise_and(img1, img2).sum()/(img1.sum(), img2.sum())

def volsim(img1, img2): # as described in https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/pdf/12880_2015_Article_68.pdf
    return 1 - (abs(img1.sum()-img2.sum())/(img1.sum()+img2.sum()))

def get_surface(img):
    return img & ~binary_erosion(img)

def hd(img1, img2, perc=95):
    coords_a = np.argwhere(get_surface(img1))
    coords_b = np.argwhere(get_surface(img2))
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)

    dist_a_to_b, _ = tree_b.query(coords_a, k=1, workers=-1)
    dist_b_to_a, _ = tree_a.query(coords_b, k=1, workers=-1)

    all_dists = np.concatenate([dist_a_to_b, dist_b_to_a])
    return np.percentile(all_dists, perc)

def evaluation_function(predictions, filename, tp_threshold=0.3): # numpy array of the predictions and the filename 
    pred_cc, pred_n_aneu = label(predictions)
    gts = load_gt(filename)
    gt_cc, gt_n_aneu = label(gts)
    
    results_per_pred_aneu = []
    for pred_obj in range(1, pred_n_aneu+1):
        results = {}
        pred_cls = np.median(predictions(pred_cc==pred_obj))
        ## find best match, if any
        best_overlap = 0
        best_overlap_id = 0
        for gt_obj in range(1, gt_n_aneu+1):
            iou = iou(gt_cc==gt_obj, pred_cc==pred_obj)
            if iou > best_overlap and iou > tp_threshold:
                best_overlap = iou
                best_overlap_id = gt_obj
                
        ## case when the prediction did not match any object in gt
        if best_overlap < tp_threshold:
            for cls in range(1, 51):
                results[f"DICE_{cls}"] = 0
                results[f"HD95_{cls}"] = 0
                results[f"VOLSIM_{cls}"] = 0
                results[f"TP_{cls}"] = 0
                results[f"FP_{cls}"] = 1 if cls == pred_cls else 0
                results[f"FN_{cls}"] = 0
                results[f"TN_{cls}"] = 0
        
        ## case when there is a match
        else:
            dsc = dice(gt_cc==best_overlap_id, pred_cc==pred_obj)
            hd95 = hd(gt_cc==best_overlap_id, pred_cc==pred_obj, 95)
            vs = volsim(gt_cc==best_overlap_id, pred_cc==pred_obj)
            gt_cls = np.median(predictions(gt_cc==best_overlap_id))
            
            for cls in range(1, 51):
                results[f"DICE_{cls}"] = dsc if cls == pred_cls and cls == gt_cls else 0
                results[f"HD95_{cls}"] = hd95 if cls == pred_cls and cls == gt_cls else 0
                results[f"VOLSIM_{cls}"] = vs if cls == pred_cls and cls == gt_cls else 0
                results[f"TP_{cls}"] = 1 if cls == pred_cls and cls == gt_cls else 0
                results[f"FP_{cls}"] = 1 if cls == pred_cls and cls != gt_cls else 0
                results[f"FN_{cls}"] = 1 if cls != pred_cls and cls == gt_cls else 0
                results[f"TN_{cls}"] = 1 if cls != pred_cls and cls != gt_cls else 0
            
        results_per_pred_aneu.append(results)
    return results_per_pred_aneu

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
    averages = {"PRECISION": 0, "RECALL":0, "MCC":0, "DICE":0, "HD95":0, "VOLSIM":0}
    for i in range(1, 51):
        averages["PRECISION"]+=metrics[f"PRECISION_{i}"]
        averages["RECALL"]+=metrics[f"RECALL_{i}"]
        averages["MCC"]+=metrics[f"MCC_{i}"]
        averages["DICE"]+=metrics[f"DICE_{i}"]
        averages["HD95"]+=metrics[f"HD95_{i}"]
        averages["VOLSIM"]+=metrics[f"VOLSIM_{i}"]
    return {k:v/50 for k, v in averages.items()}