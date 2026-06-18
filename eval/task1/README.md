# TopAneu-26 Task 1 evaluation methodology

Task 1 is an image classification task. Expected outputs are .json files with the detected aneurysm locations. Evaluation metrics are **Precision**, **Recall** and **Matthews Correlation Coefficient (MCC)** and computed for each class. Submissions are ranked by the average of these metrics across all classes.

## Method
At evaluation time all TP, FP, FN and TN are accumulated for each individual class over every prediction. TNs are counted as the number of predictions in the ground truth that is not of the currently observed class.

**Example**:

If there are the *possible classes*=[A, B, C] with *gt*=[A, B] and *predictions*=[A, C] that would result in:

- A: TP=1, FP=0, FN=0, TN=1
- B: TP=0, FP=0, FN=1, TN=1
- C: TP=0, FP=1, FN=0, TN=1

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
\text{MCC} = \frac{TN*TP-FN*FP}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

## Ranking
Submissions are ranked by first computing the average of these metrics across all classes and then by the average of the resulting global Precision, Recall and MCC.