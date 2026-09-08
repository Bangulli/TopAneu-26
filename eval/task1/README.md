# TopAneu-26 Task 1 Evaluation

Task 1 is a multi-label multi-class image classification task. Expected outputs are json files with the detected aneurysm location classes. Evaluation metrics are **Precision**, **Recall**, **F1**, and **Matthews Correlation Coefficient (MCC)** for each class. Submissions are ranked by the average of these metrics across all classes, ignoring NaN values.

## Method

Detection counts (TP, TN, FP, FN) are summed across images for each location class.
Then, precision, recall, F1, and MCC are computed from the aggregated counts for each class (division-by-zero returns NaN).

**Example**:

For these *possible classes*={A, B, C, D}, a case with *gt*=[A, B] and *predictions*=[A, C] would result in:

- A: TP=1, FP=0, FN=0, TN=0
- B: TP=0, FP=0, FN=1, TN=0
- C: TP=0, FP=1, FN=0, TN=0
- D: TP=0, FP=0, FN=0, TN=1

## Metrics

**Precision** is computed for every class using:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall** is computed for every class using:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1** is computed for every class using:

$$
\text{F1} = \frac{2 \times TP}{2 \times TP + FP + FN}
$$

**MCC** is computed for every class using:

$$
\text{MCC} = \frac{TN \times TP-FN \times FP}{\sqrt{(TP+FP)\times(TP+FN)\times(TN+FP)\times(TN+FN)}}
$$

Undefined division-by-zero results are excluded from the macro-averages.

## Ranking

For each metric, values are first averaged across images for each class. The resulting per-class averages are then averaged across all classes. Submissions are then ranked according to the average rank across these class-averaged metrics.

## Usage

### Folders `ground-truth/` and `predictions/`

When not in docker environment, you can put the prediction and ground-truth files in two subdirectories
`predictions/` and `ground-truth/` in the current directory and call `main.py` to get the evaluation results:

```sh
# mkdir and put your gt, pred like this:
├── ground-truth
├── predictions
├── main.py
```

_You can also point to any folder containing `ground-truth` and `predictions` subdirectories with the `--base_path` flag:_

```sh
# create a virtual env
python3.10 -m venv .venv
# activate
source .venv/bin/activate
# and install the dependencies (including topbrain25_eval)
pip install -r requirements.txt

# Outside of Docker, by default, main.py looks for files to evaluate in the current directory:
# in ./predictions/ and ./ground-truth/
python3 main.py

# You can override this with any folder containing the gt and pred sub-folders
python3 main.py --base_path <parent_dir_of_gt_pred_subdirs>
```

**The naming of gt and pred files can be arbitrary as long as their filenames are sorted in the same order.**

The evaluation results are saved as a JSON file at **`<parent_dir_of_gt_pred_subdirs>/output/metrics.json`**.

### Docker for GC

Only for the organizers: the bash scripts in the repo are for setting up the repo as the evaluation Docker for evaluation on Grand-challenge (GC) platform.

## Testing

We provide test cases in [test_evaluations/](test_evaluations/) to document the code and verify its correctness. Run them from the project root using `pytest`:

```sh
python3 -m pytest test_evaluations/
```
