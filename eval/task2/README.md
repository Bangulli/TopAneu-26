# TopAneu-26 Task 2 evaluation methodology

Task 2 is an instance segmentation task. Expected outputs are .mha volumes with the detected aneurysm locations. Evaluation metrics **Precision**, **Recall**, **Matthews Correlation Coefficient (MCC)**, **Dice Score (DSC)**, **Volumetric Similarity (VS)** and **Hausdorff Distance 95th precentile (HD95)** are computed for each class. Submissions are ranked by the average of these metrics across all classes. Segmentation metrics (DSC, VS, HD95) are only computed when either the GT or the Prediction contains instances of the respective class. To compute the global DSC, VS and HD95 within a given class it is accumulated over all samples and then normalized by the total count of TN, FN and FP in that class.

## Method
At evaluation time all TP, TN, FP, FN are computed for each class to obtain the classification evaluation. For that the unique values in both the GT and Prediction masks are extracted. Co-occurance means TP, Missingness in either means FP/FN, TN is the absolute amount of other-class items in the GT.

Segmentation metrics are computed on a **volume** level, disregarding connected components. The GT and Predictions are binarized for each class and the metrics are computed. **NOTE** Hausdorff distance is normalized by the diagonal of the image to make it more comparable, in case there is a FN/FP the HD is set to the max possible value (=1 since we normalize by diagonal).

**Example**:

If there are the *possible classes*=[A, B, C] with this segmentation layout:

<p align="center" width="100%">
    <img width="100%" src="fig.png"> 
</p>
That would result in:

- A: TP=1, FP=0, FN=0, TN=1
- B: TP=1, FP=0, FN=0, TN=1
- C: TP=0, FP=1, FN=0, TN=2

### Why are we not working on the instance level?
**In short**: to avoid ambiguity in edge cases. 
Evaluating on the instance level is feasible but needs many rules that make it less robust. For example if there is one big object overlapped with multiple small ones not all of the small ones can be TPs as that would inflate metrics. This is avoided by evaluating on the volume level.

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
\text{MCC} = \frac{TN \times TP - FN \times FP}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

**DSC** is computed for every class using:

$$
\text{DSC} = \frac{2 \times \lvert A \cap B \rvert}{\lvert A \rvert+\lvert B \rvert}
$$

**VS** is computed for every class using:

$$
\text{VS} = 1-\frac{\lvert \lvert A \rvert-\lvert B \rvert\rvert}{\lvert A\rvert+\lvert B\rvert}
$$

**HD95** is computed for every class as the:

95th percentile of the longest shortest bidirectional distance between two objects' surfaces.

## Ranking
Submissions are ranked by first computing the average of these metrics across all classes and then by the average of the resulting global Precision, Recall and MCC.

## Simulations
The evaluation was tested with simulated data where random images with random spheres were generated and then evaluated under different prediciton conditions. These results as well as the script can be found in [test_evaluations](test_evaluations/).

**Experiments**

- all_correct: perfect results
  - Precision: 1.00
  - Recall: 1.00
  - MCC: 1.00
  - DSC: 1.00
  - HD95: 0.00
  - VS: 1.00
- all_zero: no results detected at all
  - Precision: 0.00
  - Recall: 0.00
  - MCC: 0.00
  - DSC: 0.00
  - HD95: 1.00
  - VS: 0.00
- all_one: results at matching locations but with different shapes and all predicted labels are 1
  - Precision: 0.00
  - Recall: 0.02
  - MCC: 0.00
  - DSC: 0.00
  - HD95: 1.00
  - VS: 0.00
- 50random50correct: 50% of the samples are perfect results the other are randomly placed spheres with random size and classes in the predictions
  - Precision: 0.57
  - Recall: 0.54
  - MCC: 0.54
  - DSC: 0.38
  - HD95: 0.62
  - VS: 0.38
- random-total: All predictions are randomly placed spheres with random size and classes
  - Precision: 0.04
  - Recall: 0.04
  - MCC: 0.03
  - DSC: 0.00
  - HD95: 0.99
  - VS: 0.01
- random-ps-rv: Perfect segmentations with random labels in every object
  - Precision: 0.03
  - Recall: 0.03
  - MCC: 0.01
  - DSC: 0.01
  - HD95: 0.99
  - VS: 0.01
- random-is-pv: Segmentations with random shapes at similar locations with perfectly correct labels
  - Precision: 1.00
  - Recall: 1.00
  - MCC: 1.00
  - DSC: 0.77
  - HD95: 0.01
  - VS: 0.77
- random-is-rv: Segmentations with random shapes at similar locations with random labels
  - Precision: 0.06
  - Recall: 0.06
  - MCC: 0.05
  - DSC: 0.01
  - HD95: 0.98
  - VS: 0.02
