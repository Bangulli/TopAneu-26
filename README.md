# 📦 TopAneu-26 Baselines, Submission Templates and Evaluation
Thank you to the Grand Challenge Team for providing the basis of this repository!

## Content

This repository contains templates to help you set up your submissions for the
[TopAneu-26 challenge](https://topaneu-26.grand-challenge.org/).


It contains the following:
* ️🦾 A template for _task 1_ to base your submissions on
* ️🦾 A template for _task 2_ to base your submissions on
* ️📊 A baseline for _task 1_ **WIP**
* ️📊 A baseline for _task 2_ **WIP**
* 🧮 The _evaluation methods_ used to evaluate your submissions and to generate performance
  metrics for ranking 
* 💾 The _dataset_ for the training is provided as sha256 hashes together with a convenient download script.

Please note that this is a supplementary pack to the [Grand Challenge documentation](https://grand-challenge.org/documentation/).

## Templates

Each template contains two scritps:
 - _main.py_: Contains a set of helper functions and handles data I/O and serves images to inference functions as **Numpy Arrays**
 - _inference.py_: Contains the actual inference code for you to manipulate

**NOTE** This is just a template, any container that mimics these inputs and outputs would work, however we recommend sticking to this template and only integrating your code in the _inference.py_ scripts, as _main.py_ contains many helper functions provided by GrandChallenge, that work to serve data to your implementations.

Place the requirements of your algorithms in the respective _requirements.txt_ files.
To integrate your models, you could install them as packages and import them into the _inference.py_ script and load weights or decision tree configurations from _./models/_.
Or you can put your code directly into the image, in this case remember to include your custom files in the _Dockerfile_.

The templates also come with some bash scripts to check if your container works as GrandChallenge would run them.
  - *do_build.sh*: Builds the container under the name:tag _topaneu-26-task[1/2]:latest_.
  - *do_save.sh*: Exports the container image under the name *topaneu-26-task[1/2]_YYYY-MM-DD_hh-mm-ss.tar.gz* such that you can upload it to GC.
  - *do_test_run.sh*: Builds and runs the container as GC would. **NOTE** This script is aimed at running in an environment where GPU is available. If you're in a ressource constrained environment remove ```--gpus all``` from the run command in line 90.

To use the scripts navigate to the respective template directory:
```bash
cd templates/task[1/2]
```
Then to run the scripts do:
```bash
bash [do_build/do_save/do_test_run].sh
```
## Data
The data is hosted on [SWITCHDrive](https://drive.switch.ch/index.php/s/O36U43RkChkNcHd)
The supplementary files as well as sha256 checksums for all images and masks in the dataset are provided in [topaneu_deployment](topaneu_deployment/)
You can download the data and check the integrity using this short [script](utils/download.py) that will download the dataset to *TopAneu-26/* in the repository directory:
```bash
python utils/download.py
```
**NOTE** It requires an environment with the `requests` and `tqdm` libraries installed.

## Evaluation Methods
The evaluation methods together with simulated evaluations and results are provided for each task in [eval](eval/)
Find more details of the methodology in the READMEs of the respective folder.

The TL;DR is: 
- Task one: Multiclass image location classification
  - Expected outputs: Json files containing the predicted locations. (see [Schema](json_schema.json))
  - Predicted labels (Pred) are compared to the ground truth (GT) and per-class metrics are computed for every sample and class: TP = label present in GT and Pred, FP = present in Pred not in GT, FN = Present in GT but not in Pred, TN = N labels in GT - (TP+FN).
  - The TP, FP, TN, FN are accumulated over the whole testset and Precision, Recall and MCC are computed per class.
  - For the ranking the Precision, Recall and MCC values are averaged across classes.
- Task two: Instance Segmentation
  - Expected outputs: 3D (multi-)instance location segmentation masks.
  - The GT and predicted masks are binarized for each class, a TP = IoU > 0, FN = Present in GT but not in Pred and not TP, an FP = present in Pred not in GT and not TP, TN = N labels in GT - (TP+FN).
  - Segmentation is evaluated globally for the entire volume, **not on an instance level**. The GT and Pred are binarized for every class and Dice, Volumetric Similarity (VS) and Haussdorff Distance 95th percentile (HD95) are computed per class per sample. **NOTE** In cases where there is a FP/FN segmentation the diagnoal of the volume is used as the worst possible value.

### Run Locally
Similar to the [Templates](#templates) bash scripts are provided to run the evaluation containers locally as they would be run on GC. If you want to run the evaluation locally you can prepare the data by placing the GT files in `./eval/task[1/2]/ground_truth/[location_jsons/location_masks]` and the predicted files in subdirectories in `./eval/task[1/2]/test/input/`. Notably the input subdiretories need to fit to the convention of the specific task and contain a `predictions.json` file to match the random UIDs to the GT data. For Task 1: `./eval/task1/test/input/[UID]/output/predicted-aneurysm-location.json`; For Task 2: `./eval/task2/test/input/[UID]/output/images/aneurysm-segmentation/[UID].mha`. The `predictions.json` must map the UID to the actual filename of the sample image used to predict the output and is a list of prediction entries. Take a look at the `get_predictions_entry` function in the test scripts ([task1](eval/task1/test_evaluations/test.py), [task2](eval/task2/test_evaluations/test.py)) used to check the robustness of the methods.


## Now What?
To ensure a smooth start and avoid unnecessary frustration, it helps to first establish a
successful baseline before making any significant changes to the provided examples.

### Step 1: Run the Test Scripts
Begin by using the provided test scripts to verify that the example templates work locally for both
the example algorithm and example evaluation method.

### Step 2: Develop your method
Develop your method and integrate it into the templates.

### Step 3: Save and Upload the Algorithm
After successfully running the local test script, save the algorithm image: using the save script. On the platform: [create an algorithm](https://grand-challenge.org/documentation/create-an-algorithm-page/#creating-an-algorithm-for-a-challenge) and upload the algorithm image.

### Step 4: Submit the Example Algorithm
[Submit your algorithm image](https://grand-challenge.org/documentation/automated-evaluation/) to the challenge.

By following the steps above you will gain a solid understanding of the submission and evaluation process. This approach makes it much easier to identify and resolve any issues if something goes wrong later.

## Miscellaneous

You can find which ressources are available for your container at runtime [here](https://grand-challenge.org/documentation/runtime-environment/)

---
Generated by [Grand Challenge](https://grand-challenge.org/), modified by the TopAneu team. (2f10252)
