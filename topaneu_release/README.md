# TopAneu Training Data README

Website: https://topaneu-26.grand-challenge.org/

The TopAneu data contains processed 3D angiographic brain scans and accompanying aneurysm annotations for multimodal vessel-specific intracranial aneurysm analysis.

## Data Release

The TopAneu training data has been released in two batches. The first batch of training data was released on June 15th 2026, which contained 40 MRA cases from center-2 and 58 CTA cases from center-4. On July 31st 2026, the second batch was released, which added 200 MRA cases from center-1, 47 CTA cases from center-2, four longitudinal CTA cases from center-4, and 68 MRA cases from the public center-5.

In total, the TopAneu training dataset contains 417 scans from 409 unique patients.

The center IDs correspond to the following data sources:

| Source                                      | Country / Type | Center ID | Modalities | Patients |
|:--------------------------------------------|:---------------|:----------|:-----------|---------:|
| Lausanne University Hospital (CHUV)         | Switzerland    | center-1  | MRA        |      200 |
| Geneva University Hospitals (HUG)           | Switzerland    | center-2  | CTA, MRA   |       87 |
| Mie Chuo Medical Center                     | Japan          | center-4  | CTA        |       54 |
| Public datasets: INSTED and OpenNeuro       | Public data    | center-5  | MRA        |       68 |

Data from center-3 (University Medical Center Utrecht, UMCU) are reserved for the test set and are not included in the released training data.

The center-5 data are derived from two public datasets:

- 32 cases from INSTED[1] with identifiers beginning with `0xx`. Patients with rare aneurysm locations were selected to enrich the classes in our dataset.
- 36 cases from the Lausanne TOF-MRA aneurysm cohort on OpenNeuro[2] with identifiers beginning with `4xx`.

[1] Chen, H. et al. (2024) "INSTED: Intracranial Aneurysm and Intracranial Artery Stenosis Detection and Segmentation Challenge". https://www.codabench.org/competitions/2139/

[2] Tommaso Di Noto, Guillaume Marie, Sebastien Tourbier, Yasser Alemán-Gómez, Oscar Esteban, Guillaume Saliou, Meritxell Bach Cuadra, Patric Hagmann, and Jonas Richiardi (2022). Lausanne_TOF-MRA_Aneurysm_Cohort. OpenNeuro. https://openneuro.org/datasets/ds003949/versions/1.0.1

## Contents of Released Training Data

The release folder has the following structure:

```text
topaneu/
│
├── images/
│   ├── topaneu_center2_mr_002_0000.nii.gz
│   └── ...
├── location_masks/
│   ├── topaneu_center2_mr_002.nii.gz
│   └── ...
├── location_jsons/
│   ├── topaneu_center2_mr_002.json
│   └── ...
├── type_masks/
│   ├── topaneu_center2_mr_002.nii.gz
│   └── ...
├── vessel_masks/
│   ├── topaneu_center2_mr_002.nii.gz
│   └── ...
├── location_mapping.json
├── type_mapping.json
├── vessel_mapping.json
├── CHANGELOG.txt
├── Terms_of_use.txt
└── README.md
```

The filenames for the images and labels are named in the following schema: `topaneu_{centerID}_{modality}_{patientID}`.
Center-4 data contains a few patients with longitudinal scans, and their filenames have an appended count number (unknown order).

**Data folder description:**

| Folder / File           | Description                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `images/`               | Angiographic scans in `.nii.gz` format, processed from raw DICOMs by conversion to NIfTI, defacing, reorientation to LPS+, and cropping to the brain region. |
| `location_masks/`       | Multiclass aneurysm segmentation masks. Each aneurysm voxel is assigned to an aneurysm-location class.                                                       |
| `type_masks/`           | Aneurysm type masks with three available classes: saccular, dissecting, and fusiform.                                                                        |
| `location_jsons/`       | JSON containing a list of aneurysm locations for each case, using the same aneurysm-location classes as the location mask.                                   |
| `location_mapping.json` | Mapping between aneurysm location class names and integer label values.                                                                                      |
| `type_mapping.json`     | Mapping between aneurysm type labels and integer values.                                                                                                     |
| `vessel_masks/`         | Vessel segmentation masks predicted by TopBrain organizer model.                                                                                             |
| `vessel_mapping.json`   | Mapping between vessel anatomy labels and integer values.                                                                                                    |
| `CHANGELOG.txt`         | Summary of cases and annotations updated following quality control and review.                                                                               |
| `Terms_of_use.txt`      | Terms of use for the released data.                                                                                                                          |
| `README.md`             | Description of the dataset, folder structure, labels, usage notes, and contact information.                                                                  |


*NOTE:* The vessel segmentations were predicted by an in-house model developed by the TopBrain UZH organizers. It borrows ideas and findings from the TopBrain 2025 summary, which is available on the [medRxiv preprint](https://www.medrxiv.org/content/10.64898/2026.05.28.26354312). More information on the TopBrain vessel annotation and model can be found in the [TopBrain V2 challenge](https://topbrain2026.grand-challenge.org/).

**Mapping of aneurysm location:**

The `location_masks` and `location_jsons` folders contain multiclass aneurysm annotations. Each value corresponds to an aneurysm assigned to a specific anatomical aneurysm-location class.

*NOTE:* Since the batch-2 release, we added a new aneurysm location class, M1 early bifurcation, to the middle cerebral artery (MCA) anatomy. Now MCA has four location labels: M1 trunk, M1 early bifurcation, M1-M2 junction, and distal M2M3. (The list of images with updated locations, along with other quality control updates from batch-1, can be found in `CHANGELOG.txt`.)

| Aneurysm Location Class    | Laterality | Value |
| -------------------------- | ---------- | ----: |
| 1.1 VA trunk               | Right      |     1 |
| 1.1 VA trunk               | Left       |     2 |
| 1.2 PICA trunk             | Right      |     3 |
| 1.2 PICA trunk             | Left       |     4 |
| 1.3 VA-PICA junction       | Right      |     5 |
| 1.3 VA-PICA junction       | Left       |     6 |
| 1.4 BA trunk               | NA         |     7 |
| 1.5 VA-BA junction         | NA         |     8 |
| 1.6 AICA trunk             | Right      |     9 |
| 1.6 AICA trunk             | Left       |    10 |
| 1.7 BA-AICA junction       | Right      |    11 |
| 1.7 BA-AICA junction       | Left       |    12 |
| 1.8 SCA trunk              | Right      |    13 |
| 1.8 SCA trunk              | Left       |    14 |
| 1.9 BA-SCA junction        | Right      |    15 |
| 1.9 BA-SCA junction        | Left       |    16 |
| 1.10 BA tip                | NA         |    17 |
| 2.1 P1P2                   | Right      |    18 |
| 2.1 P1P2                   | Left       |    19 |
| 2.2 P3P4                   | Right      |    20 |
| 2.2 P3P4                   | Left       |    21 |
| 3.1 ICA infraclinoid C1-C5 | Right      |    22 |
| 3.1 ICA infraclinoid C1-C5 | Left       |    23 |
| 3.2 ICA C6-OA-junction     | Right      |    24 |
| 3.2 ICA C6-OA-junction     | Left       |    25 |
| 3.3 ICA C6-nonOA           | Right      |    26 |
| 3.3 ICA C6-nonOA           | Left       |    27 |
| 3.4 ICA C7-Pcom-junction   | Right      |    28 |
| 3.4 ICA C7-Pcom-junction   | Left       |    29 |
| 3.5 ICA C7-AChA-junction   | Right      |    30 |
| 3.5 ICA C7-AChA-junction   | Left       |    31 |
| 3.6 ICA C7-nonBranch       | Right      |    32 |
| 3.6 ICA C7-nonBranch       | Left       |    33 |
| 3.7 ICA C7-terminus        | Right      |    34 |
| 3.7 ICA C7-terminus        | Left       |    35 |
| 4.1 Acom complex           | NA         |    36 |
| 4.2 A1                     | Right      |    37 |
| 4.2 A1                     | Left       |    38 |
| 4.3 A2                     | Right      |    39 |
| 4.3 A2                     | Left       |    40 |
| 4.4 A3                     | Right      |    41 |
| 4.4 A3                     | Left       |    42 |
| 4.5 Distal ACA branches    | Right      |    43 |
| 4.5 Distal ACA branches    | Left       |    44 |
| 5.1 M1 trunk               | Right      |    45 |
| 5.1 M1 trunk               | Left       |    46 |
| 5.2 M1 early bifurcation   | Right      |    47 |
| 5.2 M1 early bifurcation   | Left       |    48 |
| 5.3 M1-M2 junction         | Right      |    49 |
| 5.3 M1-M2 junction         | Left       |    50 |
| 5.4 Distal-M2M3            | Right      |    51 |
| 5.4 Distal-M2M3            | Left       |    52 |

**Mapping of vessels and aneurysm types:**

The values for the vessel masks and aneurysm-type masks are documented in the `vessel_mapping.json` and `type_mapping.json` files.

## Changelog

More than 50 cases from the batch-1 release have been updated in the latest release. The changes include correction of annotations, improved defacing, and updates on new MCA locations. Please see `CHANGELOG.txt` for details.

## Data Usage Terms

The data are provided under the following terms:

**Open use. Must provide the source. Use for commercial purposes requires permission of the data owner.**

This means:

You may use this dataset for non-commercial research purposes.
You may use this dataset for commercial purposes only after receiving prior permission from the data owner.
You must provide the source, including author, title, and link to the dataset.
By downloading the data, you agree with the terms of use.

## Contact

If you have questions or remarks, please contact the organizers:

Ruisheng Su
Eindhoven University of Technology
r.su@tue.nl

Ekaterina Golubeva
Zürcher Hochschule für Angewandte Wissenshaften
golu@zhaw.ch

Kaiyuan Yang
University of Zürich
kaiyuan.yang@uzh.ch

Lorenz Kuhn
Eindhoven University of Technology
l.a.kuhn@tue.nl

Last updated:
July 31st 2026
