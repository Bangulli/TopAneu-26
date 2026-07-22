import json, shutil, os, subprocess, random, uuid
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, generate_binary_structure

def get_object(diameter):
    arr = np.zeros((diameter, diameter, diameter), dtype=bool)
    radius = diameter / 2
    center = (diameter - 1) / 2  # e.g. for diameter=3: center=1.0

    z, y, x = np.ogrid[:diameter, :diameter, :diameter]
    dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

    arr[dist_sq <= radius**2] = True
    return arr.astype(np.uint8)


def add_obj_to_arr(arr, value=None):
    size = random.choice(range(3, 60, 2))
    obj = get_object(size)
    if value is None: lbl = random.choice(range(1,53))
    else: lbl = value
    obj *= lbl
    x, y, z = random.choice(range(0+size, 250)), random.choice(range(0+size, 250)), random.choice(range(0+size, 250))
    arr[x-size:x, y-size:y, z-size:z] = obj
    return arr

def generate_mha(fn, value="random"):
    if value == "random":
        n_aneu = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.5, 0.15, 0.1, 0.05], k=1)[0]
        virtual_image = np.zeros((250,250,250))
        for aneu in range(n_aneu):
            virtual_image = add_obj_to_arr(virtual_image, None) 
    elif value=="one":
        n_aneu = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.5, 0.15, 0.1, 0.05], k=1)[0]
        virtual_image = np.zeros((250,250,250))
        for aneu in range(n_aneu):
            virtual_image = add_obj_to_arr(virtual_image, 1)
    else:
        virtual_image = np.zeros((250,250,250))
        
    virtual_image = sitk.GetImageFromArray(virtual_image)
    sitk.WriteImage(
        virtual_image,
        fn,
        useCompression=True,
    )
    
def random_morph(img):
    margin = random.choice(range(3, 21))
    dec = random.choice([1, 2])
    if dec == 1:
        return binary_dilation(img, structure=generate_binary_structure(3, margin))
    else:
        return binary_erosion(img, structure=generate_binary_structure(3, margin))
    
def generate_mha_like(fn, ref, value="random", mutate=True):
    if value != "zero":
        cc, n = label(ref)
        virtual_image = np.zeros_like(ref)
        for i in range(1, n+1):
            struct = random_morph(cc==i) if mutate else cc==i
            struct = struct.astype(np.uint8)
            if value == "random": struct *= random.choice(range(53))
            elif value == "one": struct *= 1
            elif value == "match": struct *= np.median(ref[cc==i]).astype(np.uint8)
            virtual_image += struct
    else: virtual_image = np.zeros_like(ref)
    virtual_image = sitk.GetImageFromArray(virtual_image)
    sitk.WriteImage(
        virtual_image,
        fn,
        useCompression=True,
    )
    
def get_predictions_entry(id, reffilename, modality):
    return {
        "pk": id,
        "inputs": [
            {
                "socket": {
                    "slug": f"head-{modality}-angiography",
                    "relative_path": f"images/head-{modality}-angio",
                    "is_image_kind": True,
                    "is_panimg_kind": True,
                    "is_dicom_image_kind": False,
                    "is_json_kind": False,
                    "is_file_kind": False
                },
                "file": None,
                "image": {
                    "name": f"{reffilename.split(".")[0]}_0000.mha"
                },
                "value": None
            }
        ],
        "outputs": [
            {
                "socket": {
                    "slug": "aneurysm-segmentation",
                    "relative_path": "images/aneurysm-segmentation",
                    "example_value": {
                        "key": "value",
                        "None": None
                    },
                    "is_image_kind": False,
                    "is_panimg_kind": False,
                    "is_dicom_image_kind": False,
                    "is_json_kind": True,
                    "is_file_kind": False
                },
                "file": None,
                "image": None,
                "value": {
                    "key": "value",
                    "None": None
                }
            }
        ],
        "exec_duration": "PT22M17S",
        "invoke_duration": None,
        "status": "Succeeded"
    }

def generate_gts(n):
    if os.path.exists("../ground_truth"):
        shutil.rmtree("../ground_truth")
    os.makedirs("../ground_truth/location_masks")
    
    for i in range(1, n+1):
        modality = random.choice(["mr", "ct"])
        center = random.choice([1, 2, 3, 4])
        caseid = "%03d" % (i,)
        fn = f"topaneu_center{center}_{modality}_{caseid}"
        generate_mha(f"../ground_truth/location_masks/{fn}.mha")
    
def generate_results_all_correct():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        
        cur_img = sitk.ReadImage(gt_dir/fn)
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        sitk.WriteImage(
            cur_img,
            pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha",
            useCompression=True
        )
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_fiftyfifty():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        
        chance = random.choice([1, 2])
        if chance == 1:
            cur_img = sitk.ReadImage(gt_dir/fn)
            os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
            sitk.WriteImage(
                cur_img,
                pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha",
                useCompression=True
            )
        else:
            os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
            cur_img = sitk.ReadImage(gt_dir/fn)
            generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img))
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

def generate_results_all_random_locations():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        generate_mha(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_all_random_labels():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        cur_img = sitk.ReadImage(gt_dir/fn)
        generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img))
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_all_random_labels_no_mutation():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        cur_img = sitk.ReadImage(gt_dir/fn)
        generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img), mutate=False)
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_all_random_mutes():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        cur_img = sitk.ReadImage(gt_dir/fn)
        generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img), value="match", mutate=True)
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_all_zero():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        cur_img = sitk.ReadImage(gt_dir/fn)
        generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img), value="zero")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

def generate_results_all_one():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_masks")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output"/"images"/"aneurysm-segmentation")
        cur_img = sitk.ReadImage(gt_dir/fn)
        generate_mha_like(pred_dir/id/"output"/"images"/"aneurysm-segmentation"/f"{uuid.uuid1()}.mha", sitk.GetArrayFromImage(cur_img), value="one")
        
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
                
def do_test_run(mode=""):
    if mode == "all_correct": # simulates a perfect submission
        generate_results_all_correct()
    elif mode == "random-total": # simulates random locations and random values
        generate_results_all_random_locations()
    elif mode == "random-ps-rv": # simulates perfect segmentations but with random values
        generate_results_all_random_labels_no_mutation()
    elif mode == "random-is-rv": # simulates randomly imperfect segmentation with random values
        generate_results_all_random_labels()
    elif mode == "random-is-pv": # simulates andomly imperfect segmentation with perfect values
        generate_results_all_random_mutes()
    elif mode == "all_zero": # simulates when all masks are all zeros
        generate_results_all_zero()
    elif mode == "all_one": # simulates imperfect segmentation and all values 1
        generate_results_all_one()
    elif mode == "50random50correct": # simulates half perfect submission with the other half random locations and random values
        generate_results_fiftyfifty()
    cmd = [
        "bash", "do_test_run.sh"
    ]
    subprocess.run(cmd, cwd="/home/tue20260926/Repos/TopAneu-26/eval/task2")
    
    print(f"Testrun for mode '{mode}' concluded. Cleaning up outputs...")
    
    shutil.move("../test/output/metrics.json", f"outputs-{mode}.json")

if __name__ == "__main__":
    generate_gts(350)
    do_test_run("all_correct")
    do_test_run("random-total")
    do_test_run("random-ps-rv")
    do_test_run("random-is-rv")
    do_test_run("random-is-pv")
    do_test_run("all_zero")
    do_test_run("all_one")
    do_test_run("50random50correct")