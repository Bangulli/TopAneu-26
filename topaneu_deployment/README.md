# TopAneu Training Data README

Website: https://topaneu-26.grand-challenge.org/

The TopAneu data contains processed 3D angiographic brain scans and accompanying aneurysm annotations for multimodal vessel-specific intracranial aneurysm analysis.

## Data Release

The TopAneu training data is released in two batches.

Batch 1 data is released on June 15th 2026. It contains 98 cases from two centers (40 MRA from center-2 and 58 CTA cases from center-4).
The center ID corresponds to the following data sources:

| Source                                  | Country / Type           | Center ID |
| --------------------------------------- | ------------------------ | --------- |
| Geneva University Hospitals, HUG        | Switzerland              | center-2  |
| Mie Chuo Medical Center                 | Japan                    | center-4  |

Batch 2 data is scheduled to be released later in July 2026 and will include data from more centers.


## Contents of Released Training Data

The release folder has the following structure:

```
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
├── Terms_of_use.txt
└── README.md
```

The filenames for the images and labels are named in the following schema: topaneu_{center}_{modality}_{caseId}.

**Data folder description:**


| Folder / File           | Description                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `images/`               | Angiographic scans in `.nii.gz` format, processed from raw DICOMs by conversion to NIfTI, defacing, reorientation to LPS+, and cropping to the brain region. |
| `location_masks/`       | Multiclass aneurysm segmentation masks. Each aneurysm voxel is assigned to a aneurysm-location class.                            |
| `type_masks/`           | Aneurysm type masks with three available classes: saccular, dissecting, and fusiform.                                          |
| `location_jsons/`       | JSON containing a list of aneurysm locations for each case, using the same aneurysm-location classes as the location_mask.                                                                                                               |
| `location_mapping.json` | Mapping between aneurysm location class names and integer label values.                                                                                        |
| `type_mapping.json`     | Mapping between aneurysm type labels and integer values.                                                                                                     |
| `vessel_masks/`         | Vessel segmentation masks predicted by TopBrain models.
|
| `vessel_mapping.json`   | Mapping between vessel anatomy labels and integer values.                                                                                                    |
| `Terms_of_use.txt`      | Terms of use for the released data.                                                                                                                          |
| `README.md`             | Description of the dataset, folder structure, labels, usage notes, and contact information.                                                                  |

**Mapping of aneurysm location:**

The location_masks and location_jsons folders contain multiclass aneurysm annotations.

Each value corresponds an aneurysm assigned to a specific anatomical aneurysm-location class.

| Aneurysm Location Class    | Laterality |      Value |
| -------------------------- | ---------- | ---------: |
| 1.1 VA trunk               | Right      |          1 |
| 1.1 VA trunk               | Left       |          2 |
| 1.2 PICA trunk             | Right      |          3 |
| 1.2 PICA trunk             | Left       |          4 |
| 1.3 VA-PICA junction       | Right      |          5 |
| 1.3 VA-PICA junction       | Left       |          6 |
| 1.4 BA trunk               | NA         |          7 |
| 1.5 VA-BA junction         | NA         |          8 |
| 1.6 AICA trunk             | Right      |          9 |
| 1.6 AICA trunk             | Left       |         10 |
| 1.7 BA-AICA junction       | Right      |         11 |
| 1.7 BA-AICA junction       | Left       |         12 |
| 1.8 SCA trunk              | Right      |         13 |
| 1.8 SCA trunk              | Left       |         14 |
| 1.9 BA-SCA junction        | Right      |         15 |
| 1.9 BA-SCA junction        | Left       |         16 |
| 1.10 BA tip                | NA         |         17 |
| 2.1 P1P2                   | Right      |         18 |
| 2.1 P1P2                   | Left       |         19 |
| 2.2 P3P4                   | Right      |         20 |
| 2.2 P3P4                   | Left       |         21 |
| 3.1 ICA infraclinoid C1-C5 | Right      |         22 |
| 3.1 ICA infraclinoid C1-C5 | Left       |         23 |
| 3.2 ICA C6-OA-junction     | Right      |         24 |
| 3.2 ICA C6-OA-junction     | Left       |         25 |
| 3.3 ICA C6-nonOA           | Right      |         26 |
| 3.3 ICA C6-nonOA           | Left       |         27 |
| 3.4 ICA C7-Pcom-junction   | Right      |         28 |
| 3.4 ICA C7-Pcom-junction   | Left       |         29 |
| 3.5 ICA C7-AChA-junction   | Right      |         30 |
| 3.5 ICA C7-AChA-junction   | Left       |         31 |
| 3.6 ICA C7-nonBranch       | Right      |         32 |
| 3.6 ICA C7-nonBranch       | Left       |         33 |
| 3.7 ICA C7-terminus        | Right      |         34 |
| 3.7 ICA C7-terminus        | Left       |         35 |
| 4.1 Acom complex           | NA         |         36 |
| 4.2 A1                     | Right      |         37 |
| 4.2 A1                     | Left       |         38 |
| 4.3 A2                     | Right      |         39 |
| 4.3 A2                     | Left       |         40 |
| 4.4 A3                     | Right      |         41 |
| 4.4 A3                     | Left       |         42 |
| 4.5 Distal ACA branches    | Right      |         43 |
| 4.5 Distal ACA branches    | Left       |         44 |
| 5.1 M1 trunk               | Right      |         45 |
| 5.1 M1 trunk               | Left       |         46 |
| 5.2 M1-M2 junction         | Right      |         47 |
| 5.2 M1-M2 junction         | Left       |         48 |
| 5.3 Distal-M2M3            | Right      |         49 |
| 5.3 Distal-M2M3            | Left       |         50 |


**Mapping of vessels and aneurysm types:**

The values for the vessel masks and aneurysm-type masks are documented in vessel_mapping.json and type_mapping.json files.

##Data Usage Terms

The data are provided under the following terms:

**Open use. Must provide the source. Use for commercial purposes requires permission of the data owner.**

This means:

You may use this dataset for non-commercial research purposes.  
You may use this dataset for commercial purposes only after receiving prior permission from the data owner.  
You must provide the source, including author, title, and link to the dataset.  
By downloading the data, you agree with the terms of use.

##Contact

If you have questions or remarks, please contact the organizers : 

Ruisheng Su
Eindhoven University of Technology
r.su@tue.nl

Ekaterina Golubeva 
Zürcher Hochschule für Angewandte Wissenshaften 
golu@zhaw.ch

Last updated:
June 15th 2026
