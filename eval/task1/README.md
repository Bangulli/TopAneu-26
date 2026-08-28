# TopAneu-26 Task 1 evaluation methodology

Task 1 is a multi-label multi-class image classification task. Expected outputs are json files with the detected aneurysm location classes. Evaluation metrics are **Precision**, **Recall** and **Matthews Correlation Coefficient (MCC)** and computed for each class. Submissions are ranked by the average of these metrics across all classes.

## Method

At evaluation time all TP, FP, FN and TN are accumulated for each individual class over every prediction.

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

**MCC** is computed for every class using:

$$
\text{MCC} = \frac{TN \times TP-FN \times FP}{\sqrt{(TP+FP)\times(TP+FN)\times(TN+FP)\times(TN+FN)}}
$$

Undefined division-by-zero results are excluded from the macro-averages.

## Ranking

Submissions are ranked by first computing the average of these metrics across all classes and then by the average rank of the resulting global Precision, Recall and MCC.

## Usage

### Folders `ground-truth/` and `predictions/`

When not in docker environment, you can put the predictions and ground-truth files into two sub-folders
`predictions/` and `ground-truth/` in the current directory and call `main.py` to get the evaluation results:

```sh
# mkdir and put your gt, pred like this:
├── ground-truth
├── predictions
├── main.py
```

_You can also specify your own custom paths for the ground-truth, predictions folders with the `--base_path` flag:_

```sh
python3 main.py

# Outside of Docker, by default, main.py looks for files to evaluate in the current directory:
# in ./predictions/ and ./ground-truth/
# You can override this with any folder containing those sub-folders
python3 main.py --base_path <path to dir with the two sub-dirs>
```

**The naming of gt and pred files can be arbitrary as long as their filenames are sorted in the same way.**

### Docker for GC

Only for the organizers: the bash scripts in the repo are for setting up the repo as the evaluation Docker for evaluation on Grand-challenge (GC) platform.

## Testing

We provide test cases in [test_evaluations/](test_evaluations/) to document the code and verify its correctness. Run them from the project root using `pytest`:

```sh
python3 -m pytest test_evaluations/
```
