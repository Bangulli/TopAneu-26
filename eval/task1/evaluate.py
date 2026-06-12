import numpy as np
import json, os
from pathlib import Path

def evaluation_function(predictions, filename):
    print(predictions, "for file", filename)
    # TODO
    return {"metrics":0}

    """ # Fourthly, load your ground truth

    # Your ground truth will be extracted to the `ground_truth_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./ground_truth` directory is mounted here
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Admin > Phase Settings and upload it under Ground Truths.
    ground_truth_dir = Path("/opt/ml/input/data/ground_truth")
    with open(
        ground_truth_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        truth = f.read()
    report += truth

    logger.info(report)

    # TODO: compare the results to your ground truth and compute some metrics

    # For now, we will just report back some bogus metric
    return {
        "my_metric": random.choice([1, 0]),
    }
    """