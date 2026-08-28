# 📦 TopAneu-26 Baselines, Submission Templates and Evaluation

Thank you to the Grand Challenge Team for providing the basis of this repository!

## Contents

This repository contains templates to help you set up your submissions for the
[TopAneu-26 challenge](https://topaneu-26.grand-challenge.org/).

It contains the following:
* ️🦾 A template for _task 1_ to base your submissions on
* ️🦾 A template for _task 2_ to base your submissions on
* 🧮 The _evaluation methods_ used to evaluate your submissions and to generate performance
  metrics for ranking 
* 💾 The _dataset_ for the training is provided as sha256 hashes together with a convenient download script.

## Templates

Each template contains two scripts:
 - _main.py_: Contains a set of helper functions and handles data I/O and serves images to inference functions as **SimpleITK image objects**
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

### Expected container outputs

#### Task1: Multiclass aneurysm location classification

A json file containing the detected locations of the sample, formatted seen in the [Schema](Task1_output_json_schema.json).
  * **NOTE** This is different from the json format the training data is provided as in `location_jsons` due to compatibility with existing Grand Challenge sockets. 
  * **NOTE** The order of the values does not matter, though for human readability ascending order is recommended.

#### Task2 Multiclass aneurysm segmentation

A 3D mask containing the multi-class aneurysm segmentations as provided in the `location_masks` in the training data.

## Data

The data is hosted on [SWITCHDrive](https://drive.switch.ch/index.php/s/O36U43RkChkNcHd)
The supplementary files as well as sha256 checksums for all images and masks in the dataset are provided in [topaneu_release](topaneu_release/)
You can download the data and check the integrity using this short [script](utils/download.py) that will download the dataset to *TopAneu-26/* in the repository directory:
```bash
python utils/download.py
```
**NOTE** It requires an environment with the `requests` and `tqdm` libraries installed.

## Evaluation Methods

The evaluation methods together with test cases are provided for each task in [eval](eval/)
Find more details of the methodology in the READMEs of the respective folder.

- [Task1 README](eval/task1/README.md):
  - Predicted labels (Pred) are compared to the ground truth (GT) and per-class metrics are computed for every sample and class.
  - The TP, FP, TN, FN are accumulated over the whole testset and Precision, Recall and MCC are computed per class.
  - For the ranking the Precision, Recall and MCC values are averaged across classes.
- [Task2 README](eval/task2/README.md):
  - The GT and predicted masks are binarized for each class, with detection determined based on overlap-threshold (non-zero overlap).
  - Segmentation is evaluated globally for the entire volume. The GT and Pred are binarized for every class and Dice, Volumetric Similarity (VS) and Haussdorff Distance 95th percentile (HD95) are computed per class per sample. **NOTE** In cases where there is a FP/FN segmentation the diagnoal of the volume is used as the worst possible value for HD95.

### Evaluate Locally

For local evaluation, you can put the predictions and ground-truth files into two sub-folders
`predictions/` and `ground-truth/` in any directory and call `main.py` using `--base_path` flag to get the evaluation results:

```sh
# from eval/task1or2
python3 main.py --base_path <path to dir containing the two sub-dirs>
```

Note: The naming of gt and pred files can be arbitrary as long as their filenames are sorted in the same way.

## Now What?

### Step 1: Develop your algorithm for either task

Develop your method and integrate it into the [Templates](./templates/).

### Step 2: Evaluate your algorithm

Evaluate locally the performance from [eval](./eval/).

### Step 3: Submit the algorithm

After successfully running the local test script, save the algorithm image: using the save script. On the platform: [create an algorithm](https://grand-challenge.org/documentation/create-an-algorithm-page/#creating-an-algorithm-for-a-challenge) and upload the algorithm image.

[Submit your algorithm image](https://topaneu-26.grand-challenge.org/evaluation/preliminary-docker-evaluation-sanity-check/submissions/create/) to the challenge.

## Extra resources

You can find which resources are available for your container at runtime [here](https://grand-challenge.org/documentation/runtime-environment/)
If you have further questions about how to set up your container, or if you do not want to follow this template you can find more information [here](https://grand-challenge.org/documentation/algorithms/).

---
Generated by [Grand Challenge](https://grand-challenge.org/), modified by the TopAneu team. (2f10252)
