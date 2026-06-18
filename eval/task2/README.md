# TopAneu-26 Task 2 evaluation methodology

Task 1 is an image classification task. Expected outputs are .json files with the detected aneurysm locations. Evaluation metrics **Precision**, **Recall**, **Matthews Correlation Coefficient (MCC)**, **Dice Score (DSC)**, **Volumetric Similarity (VS)** and **Hausdorff Distance 95th precentile (HD95)** are computed for each class. Submissions are ranked by the average of these metrics across all classes. Segmentation metrics (DSC, VS, HD95) are only computet on TPs.

## Method
At evaluation time all TP, FP, FN and TN are accumulated for each individual class over every prediction. Segmentations are counted as TP if they have the same class and more than 0.3 DSC with the corresponding GT object. TNs are counted as the number of predictions in the ground truth that is not of the currently observed class.

**Example**:

If there are the *possible classes*=[A, B, C] with this segmentation layout:

<p align="center" width="100%">
    <img width="100%" src="fig.png"> 
</p>
That would result in:

- A: TP=1, FP=0, FN=0, TN=1
- B: TP=0, FP=1, FN=1, TN=1
- C: TP=0, FP=1, FN=0, TN=2

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

**DSC** is computed for every class using:
$$
\text{DSC} = \frac{2*\lvert A \cap B \rvert}{\lvert A \rvert+\lvert B \rvert}
$$

**VS** is computed for every class using:
$$
\text{VS} = 1-\frac{\lvert \lvert A \rvert-\lvert B \rvert\rvert}{\lvert A\rvert+\lvert B\rvert}
$$

**HD95** is computed for every class by:

- Subtracting the erosion of the object from the object to obtain the surface voxels.
- Finding the minium euclidean distance for each surface voxel in A to each surface voxel in B.
- Computing the 95th percentile of the resulting list of minimum distances.

## Ranking
Submissions are ranked by first computing the average of these metrics across all classes and then by the average of the resulting global Precision, Recall and MCC.