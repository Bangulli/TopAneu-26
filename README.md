# 📦 TopAneu-26 Baselines, Submission Templates and Evaluation

Thank you to the Grand Challenge Team for providing the basis of this repository!

## Contents

This repository contains templates to help you set up your submissions for the
[TopAneu-26 challenge](https://topaneu-26.grand-challenge.org/).

It contains the following:
* ️🦾 A template for _Task 1_ to base your submission on
* ️🦾 A template for _Task 2_ to base your submission on
* 🧮 The _evaluation methods_ used to evaluate your submissions and generate performance metrics for ranking

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

#### Task 1: Multiclass aneurysm location classification

A json file containing the detected locations of the sample, formatted seen in the [schema](Task1_output_json_schema.json).
  * **NOTE** This is different from the json format the training data is provided as in `location_jsons` due to compatibility with existing Grand Challenge sockets. 
  * **NOTE** The order of the values does not matter, though for human readability ascending order is recommended.

#### Task 2: Multiclass aneurysm segmentation

A 3D mask containing the multi-class aneurysm segmentations as provided in the `location_masks` in the training data.

## Evaluation

The evaluation methods, together with testing and documentation, are provided for each task in the [eval/](eval/) folder.
For more details on the methodology, see the README in the respective task folder.

- [Task 1 Evaluation README](eval/task1/README.md)
- [Task 2 Evaluation README](eval/task2/README.md)

### Evaluate Locally

For local evaluation, you can put the prediction and ground-truth files in two subdirectories
`predictions/` and `ground-truth/`, **under the same parent directory**.
Then, run `main.py` with the `--base_path` flag to get the evaluation results:

```sh
# from eval/task1 or eval/task2
python3 main.py --base_path <parent_dir_of_gt_pred_subdirs>

# example usage
cd eval/task1 # or cd eval/task2
python3 main.py --base_path ./test_evaluations/
```

Note: The naming of gt and pred files can be arbitrary as long as their filenames are sorted in the same order.

The evaluation results are saved as a JSON file at **`<parent_dir_of_gt_pred_subdirs>/output/metrics.json`**.

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
