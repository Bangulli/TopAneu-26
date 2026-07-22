import shutil, os, json
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from pprint import pprint

def get_image_info(img: sitk.Image) -> dict:
    return {
        "shape": img.GetSize(),          # (x, y, z)
        "spacing": img.GetSpacing(),      # (sx, sy, sz)
        "origin": img.GetOrigin(),        # (ox, oy, oz)
        "orientation": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection()) # 9-element direction cosine matrix (flattened)
    }

def check_topaneu_dir(dir, ignore_vessels=True):
    dir = Path(dir)
    assert (dir/"images").exists() and (dir/"location_masks").exists() and (dir/"location_jsons").exists() and (dir/"type_masks").exists() and (dir/"vessel_masks").exists(), "Not even all necessary directories are present"
    if os.path.exists(dir/"qc_report.json"):
        with open(dir/"qc_report.json", "r") as f:
            report=json.load(f)
        images = list(report["action_needed"].keys())
        if any(images):
            print("######## Rerunning Checks for cases ########")
            pprint(report["action_needed"])
            report["action_needed"]={}
        else:
            print(f"######## already checked all cases for dir {dir}, no actions needed. ########")
            return
    
    else:
        report = {}
        report["directory"] = str(dir)
        report["action_needed"] = {}
        images = os.listdir(dir/"images")
    
    
    jsons = os.listdir(dir/"location_jsons")
    l_masks = os.listdir(dir/"location_masks")
    t_maks = os.listdir(dir/"type_masks")
    if not ignore_vessels: v_maks = os.listdir(dir/"vessel_masks") 
    
    for img in tqdm(images, desc="checking..."):
        try:
            cur_report = {}
            cur_json = None
            cur_img_info = get_image_info(sitk.ReadImage(dir/"images"/img))
            if cur_img_info["orientation"] == "LPS": cur_report["is_LPS"] = True
            else: cur_report["is_LPS"] = False; ok = False
            
            for k, v in cur_img_info.items(): cur_report[k]=v
            
            cur_lmask = None
            cur_tmask = None
            cur_vmask = None
            ok = True
            fn_msk = img.replace("_0000.", ".")
            fn_jsn = fn_msk.replace(".nii.gz", ".json")

            ## Check existance of files
            if fn_jsn in jsons: 
                cur_report["has_json"] = True
                with open(dir/"location_jsons"/fn_jsn, "r") as f:
                    cur_json = json.load(f)
            else: cur_report["has_json"] = False; ok = False

            if fn_msk in l_masks: 
                cur_report["has_lmask"] = True
                cur_lmask = sitk.ReadImage(dir/"location_masks"/fn_msk)
            else: cur_report["has_lmask"] = False; ok = False
            
            if fn_msk in t_maks: 
                cur_report["has_tmask"] = True
                cur_tmask = sitk.ReadImage(dir/"type_masks"/fn_msk)
            else: cur_report["has_tmask"] = False; ok = False
            
            if not ignore_vessels:
                if fn_msk in v_maks: 
                    cur_report["has_vmask"] = True
                    cur_vmask = sitk.ReadImage(dir/"vessel_masks"/fn_msk)
                else: cur_report["has_vmask"] = False; ok = False
            
            ## Check correspondence of labels
            if cur_lmask is not None:
                lbls = cur_json["locations"]
                lbls.sort()
                if lbls == [lbl for lbl in np.unique(sitk.GetArrayFromImage(cur_lmask)).tolist() if lbl != 0]:
                    cur_report["labels_match"] = True
                else: cur_report["labels_match"] = False; ok = False
                    
            ## Check correspondence of mask shape
            if cur_lmask is not None:
                cur_lmask_info = get_image_info(cur_lmask)
                for k in cur_img_info.keys():
                    img_v = cur_img_info[k]
                    msk_v = cur_lmask_info[k]
                    if img_v == msk_v: cur_report[f"{k}s_match_lmask"]=True
                    else: cur_report[f"{k}s_match_lmask"]=False; ok=False
                    
            if cur_tmask is not None:
                cur_tmask_info = get_image_info(cur_tmask)
                for k in cur_img_info.keys():
                    img_v = cur_img_info[k]
                    msk_v = cur_tmask_info[k]
                    if img_v == msk_v: cur_report[f"{k}s_match_tmask"]=True
                    else: cur_report[f"{k}s_match_tmask"]=False; ok=False
            
            if not ignore_vessels:    
                if cur_vmask is not None:
                    cur_vmask_info = get_image_info(cur_vmask)
                    for k in cur_img_info.keys():
                        img_v = cur_img_info[k]
                        msk_v = cur_vmask_info[k]
                        if img_v == msk_v: cur_report[f"{k}s_match_vmask"]=True
                        else: cur_report[f"{k}s_match_vmask"]=False; ok=False
            
            if not ok: report["action_needed"][img]=cur_report
            else: report[img]=cur_report
            
        except Exception as e:
            report["action_needed"][img]=f"Error Occured: {e}"
            
    print("########### CHECK COMPLETE ###########")
    if any(report["action_needed"]): 
        print("!! Mismatches Found !!")
        pprint(report["action_needed"])
    
    with open(dir/"qc_report.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    check_topaneu_dir(Path("Datasets/TopAneu/Test"))
    check_topaneu_dir(Path("Datasets/TopAneu/Train"), ignore_vessels=False)
