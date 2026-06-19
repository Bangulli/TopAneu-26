# Grand Challenge requires to upload files in .mha format, this script does the conversion

import SimpleITK as sitk, os, pathlib as pl

def convert_directory(path):
    for f in os.listdir(path):
        img = sitk.ReadImage(path/f)
        sitk.WriteImage(img,
                        path/f.replace(".nii.gz", ".mha"),
                        useCompression=True)

if __name__=="__main__":
    convert_directory(pl.Path("/home/tue20260926/tue/Documents/Datasets/GrandChallenge-Phase1-testdata/images"))
    convert_directory(pl.Path("/home/tue20260926/tue/Documents/Datasets/GrandChallenge-Phase1-testdata/location_masks"))