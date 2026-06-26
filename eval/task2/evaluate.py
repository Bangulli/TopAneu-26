import numpy as np
import json, os, math, glob
from pathlib import Path
import SimpleITK as sitk
from typing import Union, List, Tuple, Literal
from scipy.ndimage import label, binary_erosion
from scipy.spatial import cKDTree

### copied from main
def load_image_file_as_array(location: Path) -> np.ndarray:
    """Load an image as a numpy array using sitk

    Args:
        location (Path): Path to the image

    Returns:
        np.ndarray: The image as a 3d array
    """
    result = sitk.ReadImage(location)
    # Convert it to a Numpy array
    return sitk.GetArrayFromImage(result)

def load_gt(fn: str) -> np.ndarray:
    """Load a ground truth reference image for a given filepath

    Args:
        fn (str): The filename of the image

    Returns:
        np.ndarray: The correpsonding gt segmentation as a 3d array
    """
    dir = Path("/opt/ml/input/data/ground_truth/location_masks")
    return load_image_file_as_array(dir/fn.replace("_0000.mha", ".mha"))

def iou_fn(img1: np.ndarray, img2: np.ndarray, eps: float=1e-6) -> float:
    """Compute the iou between two binary images

    Args:
        img1 (np.ndarray): Image a
        img2 (np.ndarray): Image b
        eps (float, optional): Epsilon to avoid div by zero error. Defaults to 1e-6.

    Returns:
        float: The iou between a and b
    """
    return (np.bitwise_and(img1, img2).sum()/(np.bitwise_or(img1, img2).sum())+eps)

def dice_fn(img1: np.ndarray, img2: np.ndarray, eps: float=1e-6) -> float:
    """Compute the dice score between two binary images

    Args:
        img1 (np.ndarray): Image a
        img2 (np.ndarray): Image b
        eps (float, optional): Epsilon to avoid div by zero error. Defaults to 1e-6.

    Returns:
        float: The dice score between a and b
    """
    return ((2*np.bitwise_and(img1, img2).sum())/(img1.sum() + img2.sum() + eps))

def volsim_fn(img1: np.ndarray, img2: np.ndarray, eps: float=1e-6) -> float: # as described in https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/pdf/12880_2015_Article_68.pdf
    """Compute the volumetric similarity between two binary images

    Args:
        img1 (np.ndarray): Image a
        img2 (np.ndarray): Image b
        eps (float, optional): Epsilon to avoid div by zero error. Defaults to 1e-6.

    Returns:
        float: The volumetric similarity between a and b
    """
    return 1 - (abs(img1.sum()-img2.sum())/(img1.sum()+img2.sum()+eps))

def get_surface(img: np.ndarray) -> np.ndarray:
    """Get the surface of the objects in a 3d binary image

    Args:
        img (np.ndarray): A 3d binary image

    Returns:
        np.ndarray: A 3d binary image containing only the surface of the input image
    """
    return img & ~binary_erosion(img)

def hd_fn(img1: np.ndarray, img2: np.ndarray, perc: int=95, norm: bool=True) -> float: # NOTE this function is claude generated as my implementation with a double loop was too slow.
    """Compute the percentile of the hausdorff distance between the surfaces of two objects in 3d space
        In case of either mask being nonzero with the respective other being zero, will return the diagonal of the volume

    Args:
        img1 (np.ndarray): Image a
        img2 (np.ndarray): Image b
        perc (int, optional): The percentile. Defaults to 95.
        norm (bool, optional): Whether to return absolute or normalized values. Normalized by the diagnoal of the volume. Defaults to True.

    Returns:
        float: The perc percentile of the bidirectional HD between a and b
    """
    diag = np.linalg.norm(img1.shape)
    if ~np.any(img1) and ~np.any(img2): return 0 # both 0 is perfect seg
    elif ~np.any(img1) or ~np.any(img2): return 1 if norm else diag # only one zero is the worst possible segmentation
        
    coords_a = np.argwhere(get_surface(img1))
    coords_b = np.argwhere(get_surface(img2))

    tree_b = cKDTree(coords_b)
    tree_a = cKDTree(coords_a)

    d_ab, _ = tree_b.query(coords_a)   # a -> nearest b
    d_ba, _ = tree_a.query(coords_b)   # b -> nearest a
    
    hd = max(np.percentile(d_ab, perc),
               np.percentile(d_ba, perc))

    return hd/diag if norm else hd

def extended_label(img: np.ndarray) -> Tuple[List[np.ndarray], List[int], int]:
    """Get the connected components and labels for every class in a segmentation mask

    Args:
        img (np.ndarray): The segmentation mask

    Returns:
        Tuple[List[np.ndarray], List[int], int]: The connceted components for each class, the n components for each class and the total number of components in the image.
    """
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

def check_tp(img1: np.ndarray, img2: np.ndarray, threshold: float=0.3, method: Literal["intersection", "dice", "iou"]="intersection") -> bool:
    """Check if two segmentations are a true positive

    Args:
        img1 (np.ndarray): Segmentation a
        img2 (np.ndarray): Segmentation b
        threshold (float, optional): The threshold to determin a tp. Defaults to 0.3.
        method (Literal[&quot;intersection&quot;, &quot;dice&quot;, &quot;iou&quot;], optional): The metric to consider. Defaults to "intersection".

    Raises:
        ValueError: If the passed method is not supported

    Returns:
        bool: Whether the two input segmentation fulfill the TP criterion.
    """
    match method:
        case "intersection":
            return np.bitwise_and(img1, img2).sum() > threshold
        case "dice":
            return dice_fn(img1, img2) > threshold
        case "iou":
            return iou_fn(img1, img2) > threshold
        case _:
            raise ValueError(f"Invalid tp checking method '{method}'")

def evaluation_function_instance_level(predictions: np.ndarray, filename: str) -> dict: # numpy array of the predictions and the filename 
    """Computes the amount of TP, FP, TN, FN and the Dice Score, Volumetric Similarity and HD95 for the TPs for all possible classes for a given sample.

    Args:
        predictions (np.ndarray): The predicted segmentation
        filename (str): The name of the corresponding image

    Returns:
        dict: The metrics.
    """
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
                    if check_tp(gt==j, pred==i, threshold=0, method="intersection"):
                        hd95 += hd_fn(gt==j, pred==i, 95)
                        vs += volsim_fn(gt==j, pred==i) 
                        dsc += dice_fn(gt==j, pred==i)
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

def evaluation_function(predictions: np.ndarray, filename: str) -> dict:
    """Computes the amount of TP, FP, TN, FN and the Dice Score, Volumetric Similarity and HD95 for the TPs for all possible classes for a given sample.

    Args:
        predictions (np.ndarray): The predicted segmentation
        filename (str): The name of the corresponding image

    Returns:
        dict: The metrics.
    """
    gt = load_gt(filename)
    
    results = {}
    preds = np.unique(predictions).tolist()
    gts = np.unique(gt).tolist()
    n_aneu = len(gts)

    for cls in range(1, 51): 
        pred_count = len([lbl for lbl in preds if lbl==cls])
        gt_count = len([lbl for lbl in gts if lbl==cls])
        tp = min(pred_count, gt_count)
        fp = max(0, pred_count - gt_count)
        fn = max(0, gt_count - pred_count)
        tn = n_aneu-(tp+fn)
        results[f"TP_{cls}"] = tp
        results[f"FP_{cls}"] = fp
        results[f"FN_{cls}"] = fn
        results[f"TN_{cls}"] = tn
        
        if pred_count > 0 or gt_count > 0: # only compute if necessary to save runtime. Comp is necessary if any segmentation is available
            results[f"DICE_{cls}"] = dice_fn(gt==cls, predictions==cls)
            results[f"HD95_{cls}"] = hd_fn(gt==cls, predictions==cls, 95, True)
            results[f"VOLSIM_{cls}"] = volsim_fn(gt==cls, predictions==cls)
        else:
            results[f"DICE_{cls}"] = 0
            results[f"HD95_{cls}"] = 0
            results[f"VOLSIM_{cls}"] = 0


            
        
    # results["INSTANCE_LEVEL"] = evaluation_function_instance_level(predictions, filename) omit for now.

    return results
    

def evaluation_aggregation(metrics: List[dict], eps: float=1e-6) -> dict: # epsilons to avoid div by 0 error
    """Aggregates the metrics over all samples and computes the per class Precision, Recal and MCC and normalizes the Dice Score, Volumetric Similarity and HD95 by the total amount of all TP and FN for each class.

    Args:
        metrics (List[dict]): The metrics for each sample
        eps (float, optional): The epsilon to avoid division by zero. Defaults to 1e-6.

    Returns:
        dict: The aggregated metrics for each class
    """
    aggregates = {}
    keys = metrics[0].keys()
    ## aggregate all tpfpfn
    for k in keys:
        if k == 'INSTANCE_LEVEL':continue
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
        aggregates[f"DICE_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+aggregates[f"FP_{i}"]+eps)
        aggregates[f"HD95_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+aggregates[f"FP_{i}"]+eps)
        aggregates[f"VOLSIM_{i}"] /= (aggregates[f"TP_{i}"]+aggregates[f"FN_{i}"]+aggregates[f"FP_{i}"]+eps)
    return aggregates

def evaluation_average(metrics: dict) -> dict:
    """Computes the average metrics across classes for ranking

    Args:
        metrics (dict): The per-class metrics

    Returns:
        dict: The average-across-classes metrics
    """
    averages = {"PRECISION": 0, "RECALL":0, "MCC":0, "DICE":0, "HD95":0, "VOLSIM":0}
    for i in range(1, 51):
        averages["PRECISION"]+=metrics[f"PRECISION_{i}"]
        averages["RECALL"]+=metrics[f"RECALL_{i}"]
        averages["MCC"]+=metrics[f"MCC_{i}"]
        averages["DICE"]+=metrics[f"DICE_{i}"]
        averages["HD95"]+=metrics[f"HD95_{i}"]
        averages["VOLSIM"]+=metrics[f"VOLSIM_{i}"]
    return {k:v/50 for k, v in averages.items()}
