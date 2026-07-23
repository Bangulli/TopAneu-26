"""
This is an example for an inference function for your models
"""
import numpy as np
from pathlib import Path
import SimpleITK as sitk

def infer_ct(img: sitk.Image) -> np.ndarray:
    """Runs inference on CTA images

    Args:
        img (sitk.Image): The image to predict on.

    Returns:
        sitk.Image: The generated mask.
    """
    # You can upload data to grandchallenge alongside your container, for example model weights.
    # Your model weights will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    # Here's how you can access it in the container on GC
    img_arr = sitk.GetArrayFromImage(img)
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())
                
    res_arr = np.ones_like(img_arr, dtype=np.uint8)*52
    res = sitk.GetImageFromArray(res_arr)
    res.CopyInformation(img)

    # For now, let us make bogus predictions
    return res

def infer_mr(img: sitk.Image) -> np.ndarray:
    """Runs inference on MRA images

    Args:
        img (sitk.Image): The image to predict on.

    Returns:
        sitk.Image: The generated mask.
    """
    # You can upload data to grandchallenge alongside your container, for example model weights.
    # Your model weights will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    # Here's how you can access it in the container on GC
    img_arr = sitk.GetArrayFromImage(img)
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())
    res_arr = np.ones_like(img_arr, dtype=np.uint8)*52
    res = sitk.GetImageFromArray(res_arr)
    res.CopyInformation(img)

    # For now, let us make bogus predictions
    return res