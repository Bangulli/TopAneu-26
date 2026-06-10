"""
This is an example for an inference function for your models
"""
import numpy as np
from pathlib import Path

def infer_ct(img: np.ndarray) -> np.ndarray:
    """Runs inference on CTA images

    Args:
        img (np.ndarray): The image to predict on.

    Returns:
        np.ndarray: The generated mask.
    """
    # You can upload data to grandchallenge alongside your container, for example model weights.
    # Your model weights will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    # Here's how you can access it in the container on GC
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # For now, let us make bogus predictions
    return np.zeros_like(img)

def infer_mr(img: np.ndarray) -> np.ndarray:
    """Runs inference on MRA images

    Args:
        img (np.ndarray): The image to predict on.

    Returns:
        np.ndarray: The generated mask.
    """
    # You can upload data to grandchallenge alongside your container, for example model weights.
    # Your model weights will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    # Here's how you can access it in the container on GC
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # For now, let us make bogus predictions
    return np.zeros_like(img)