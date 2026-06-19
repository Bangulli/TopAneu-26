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

def iou_fn(img1, img2, eps=1e-6):
    return (np.bitwise_and(img1, img2).sum()/(np.bitwise_or(img1, img2).sum())+eps)

def dice_fn(img1, img2, eps=1e-6):
    return ((2*np.bitwise_and(img1, img2).sum())/(img1.sum() + img2.sum() + eps))

def volsim_fn(img1, img2, eps=1e-6): # as described in https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/pdf/12880_2015_Article_68.pdf
    return 1 - (abs(img1.sum()-img2.sum())/(img1.sum()+img2.sum()+eps))

def get_surface(img):
    return img & ~binary_erosion(img)

def hd_fn(img1, img2, perc=95): # NOTE this function is claude generated as my implementation with a double loop was too slow.
    coords_a = np.argwhere(get_surface(img1))
    coords_b = np.argwhere(get_surface(img2))

    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.nan

    tree_b = cKDTree(coords_b)
    tree_a = cKDTree(coords_a)

    d_ab, _ = tree_b.query(coords_a)   # a -> nearest b
    d_ba, _ = tree_a.query(coords_b)   # b -> nearest a

    return max(np.percentile(d_ab, perc),
               np.percentile(d_ba, perc))

def extended_label(img):
    ccs = []
    ns = []
    present_cls = np.unique(img).tolist()
    for cls in range(1, 51):
        if cls not in present_cls:
            ccs.append(None)
            ns.append(0)
        else:
            cc, n = label(img==cls)
            ccs.append(cc)
            ns.append(n)
    return ccs, ns, sum(ns)

def evaluation_function(predictions, filename, tp_threshold=0.3): # numpy array of the predictions and the filename 
    gts = load_gt(filename)
    gt_ccs, gt_n_per_aneu, gt_n_aneu = extended_label(gts)
    pred_ccs, pred_n_per_aneu, pred_n_aneu = extended_label(predictions)
    
    results = {}
    for cls in range(1, 51):
        gt = gt_ccs[cls-1]
        pred = pred_ccs[cls-1]
        
        # eval seg per case
        dsc = 0
        hd95 = 0
        vs = 0

        # eval cls performance per instance
        tp = 0
        fp = 0
        if pred_n_per_aneu[cls-1]>0 and gt_n_per_aneu[cls-1]>0: # if both are nonzero do the loop and look for matches
            for i in range(1,pred_n_per_aneu[cls-1]+1):
                is_tp = False
                for j in range(1,gt_n_per_aneu[cls-1]+1):
                    cur_dsc = dice_fn(gt==j, pred==i)
                    if cur_dsc > tp_threshold:
                        hd95 -= hd_fn(gt==j, pred==i, 95)
                        vs += volsim_fn(gt==j, pred==i) 
                        dsc += cur_dsc
                        is_tp = True
                        break
                if is_tp: tp += 1
                else: fp += 1
            fn = gt_n_per_aneu[cls-1]-tp
            tn = gt_n_aneu-(tp+fn)
            
        elif pred_n_per_aneu[cls-1]>0 and gt_n_per_aneu[cls-1]==0: # if no samples of the class available in the gt but there are any in the predictions, count everything as false positive, but also add the true negatives for the other classes
            fp = pred_n_per_aneu[cls-1]
            fn = 0
            tn = gt_n_aneu
            
        elif pred_n_per_aneu[cls-1]==0 and gt_n_per_aneu[cls-1]>0: # if no samples of the class available in the predictions but there are any in the gt, count everything as false negative, but also add the true negatives for the other classes
            fn = gt_n_per_aneu[cls-1]
            tn = gt_n_aneu-fn
            
        else: # If there are no objects of a class in neither gt nor predictions, count only true negatives
            fn = 0
            tn = gt_n_aneu

        results[f"DICE_{cls}"] = dsc
        results[f"HD95_{cls}"] = hd95
        results[f"VOLSIM_{cls}"] = vs
        results[f"TP_{cls}"] = tp
        results[f"FP_{cls}"] = fp
        results[f"FN_{cls}"] = fn
        results[f"TN_{cls}"] = tn
    return results

def evaluation_aggregation(metrics, eps=1e-6): # epsilons to avoid div by 0 error
    aggregates = {}
    keys = metrics[0].keys()
    ## aggregate all tpfpfn
    for k in keys:
        aggregates[k]=sum([samp[k] for samp in metrics])
    ## compute per location prec, rec, mcc
    for i in range(1, 51):
        
        # comp cls scores
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
        ) + eps
        aggregates[f"MCC_{i}"] = mcc_num/mcc_den
        
        # comp segmentation scores
        aggregates[f"DICE_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+eps)
        aggregates[f"HD95_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+eps)
        aggregates[f"VOLSIM_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+eps)
    return aggregates

def evaluation_average(metrics):
    averages = {"PRECISION": 0, "RECALL":0, "MCC":0, "DICE":0, "HD95":0, "VOLSIM":0}
    for i in range(1, 51):
        averages["PRECISION"]+=metrics[f"PRECISION_{i}"]
        averages["RECALL"]+=metrics[f"RECALL_{i}"]
        averages["MCC"]+=metrics[f"MCC_{i}"]
        averages["DICE"]+=metrics[f"DICE_{i}"]
        averages["HD95"]+=metrics[f"HD95_{i}"]
        averages["VOLSIM"]+=metrics[f"VOLSIM_{i}"]
    return {k:v/50 for k, v in averages.items()}
