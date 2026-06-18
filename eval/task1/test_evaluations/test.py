import json, shutil, os, subprocess, random, uuid
from pathlib import Path

def generate_json(fn, value="random"):
    if value == "random":
        n_aneu = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.5, 0.15, 0.1, 0.05], k=1)[0]
        if n_aneu > 0:
            with open(fn, "w") as f:
                json.dump({"locations": random.sample(range(1,51), k=n_aneu)}, f, indent=4)
        else:
            with open(fn, "w") as f:
                json.dump({"locations": []}, f, indent=4)
    elif value=="one":
        with open(fn, "w") as f:
            json.dump({"locations": [1]}, f, indent=4)
    else:
        with open(fn, "w") as f:
            json.dump({"locations": []}, f, indent=4)
    
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
                    "name": f"{reffilename.split(".")[0]}_0000.nii.gz"
                },
                "value": None
            }
        ],
        "outputs": [
            {
                "socket": {
                    "slug": "detected-aneurysm-locations",
                    "relative_path": "detected-aneurysm-locations.json",
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
    os.makedirs("../ground_truth/location_jsons")
    
    for i in range(1, n+1):
        modality = random.choice(["mr", "ct"])
        center = random.choice([1, 2, 3, 4])
        caseid = "%03d" % (i,)
        fn = f"topaneu_center{center}_{modality}_{caseid}"
        generate_json(f"../ground_truth/location_jsons/{fn}.json")
    
def generate_results_all_correct():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_jsons")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        
        with open(gt_dir/fn, "r") as f:
            cur_preds = json.load(f)  
        os.makedirs(pred_dir/id/"output")
        with open(pred_dir/id/"output"/"detected-aneurysm-locations.json", "w") as f:
            json.dump(cur_preds, f, indent=4)
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_fiftyfifty():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_jsons")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        
        chance = random.choice([1, 2])
        if chance == 1:
            with open(gt_dir/fn, "r") as f:
                cur_preds = json.load(f)  
            os.makedirs(pred_dir/id/"output")
            with open(pred_dir/id/"output"/"detected-aneurysm-locations.json", "w") as f:
                json.dump(cur_preds, f, indent=4)
        else:
            os.makedirs(pred_dir/id/"output")
            generate_json(pred_dir/id/"output"/"detected-aneurysm-locations.json")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

def generate_results_all_random():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_jsons")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output")
        generate_json(pred_dir/id/"output"/"detected-aneurysm-locations.json")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
def generate_results_all_zero():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_jsons")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output")
        generate_json(pred_dir/id/"output"/"detected-aneurysm-locations.json", "zero")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

def generate_results_all_one():
    pred_dir = Path("../test/input")
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
    
    predictions = []
    gt_dir = Path("../ground_truth/location_jsons")
    for fn in os.listdir(gt_dir):
        modality = "mr" if "_mr_" in fn else "ct"
        id = str(uuid.uuid1())
        
        predictions.append(get_predictions_entry(id, fn, modality))
        os.makedirs(pred_dir/id/"output")
        generate_json(pred_dir/id/"output"/"detected-aneurysm-locations.json", "one")
            
    with open(pred_dir/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
                
def do_test_run(mode=""):
    if mode == "all_correct":
        generate_results_all_correct()
    elif mode == "random":
        generate_results_all_random()
    elif mode == "all_zero":
        generate_results_all_zero()
    elif mode == "all_one":
        generate_results_all_one()
    elif mode == "50random50correct":
        generate_results_fiftyfifty()
    cmd = [
        "bash", "do_test_run.sh"
    ]
    subprocess.run(cmd, cwd="/home/tue20260926/Repos/TopAneu-26/eval/task1")
    
    print(f"Testrun for mode '{mode}' concluded. Cleaning up outputs...")
    
    shutil.move("../test/output/metrics.json", f"outputs-{mode}.json")

if __name__ == "__main__":
    do_test_run("all_correct")
    do_test_run("random")
    do_test_run("all_zero")
    do_test_run("all_one")
    do_test_run("50random50correct")