"""
This is an example for an inference function for your models
"""
import numpy as np
from pathlib import Path
import random

def infer_ct(img: np.ndarray) -> list:
    """Predicts aneurysm locations in CTA images.

    Args:
        img (np.ndarray): The image to predict.

    Returns:
        list: The detected locations in a list.
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
    n_aneu = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.5, 0.15, 0.1, 0.05], k=1)[0]
    if n_aneu > 0: preds = random.sample(range(1,51), k=n_aneu)
    else: preds = []

    return preds

def infer_mr(img: np.ndarray) -> list:
    """Predicts aneurysm locations in CTA images.

    Args:
        img (np.ndarray): The image to predict.

    Returns:
        list: The detected locations in a list.
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
    n_aneu = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.5, 0.15, 0.1, 0.05], k=1)[0]
    if n_aneu > 0: preds = random.sample(range(1,51), k=n_aneu)
    else: preds = []
    return preds