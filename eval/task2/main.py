"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import glob
import json
import logging
import random
from pathlib import Path
from pprint import pformat
from statistics import mean

import SimpleITK
from helpers import run_prediction_processing, setup_logger, tree

from evaluate import evaluation_function

logger = logging.getLogger("evaluate")


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")


def main():
    setup_logger(
        # Optionally: change this to the more verbose DEBUG
        level=logging.INFO,
    )

    log_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any specific order!
    # We work that out from predictions.json

    # predictions.json contains information about each job's inputs and outputs
    # along with possible timing information. There are potentially up to two times
    # available as ISO-8601 duration strings or None if not measured.
    # We advise parsing these with isodate.parse_duration, available on pypi.
    #
    # We advise caution if you are considering using these times for ranking purposes,
    # as they may not be stable for the duration of the challenge due to changes
    # in the underlying implementation or infrastructure, or be repeatable
    # due to shared hardware issues.
    #
    # `exec_duration`
    #     The duration of the execution, **if measured**. Excludes data
    #     validation, container pulling, model downloading, data downloading
    #     and data uploading times.
    #
    #     Includes model loading time, input data loading time,
    #     processing time, output data writing time and
    #     **any delays from shared hardware issues**.
    #
    # `invoke_duration`
    #     The duration of the execution, **if measured**. Excludes data
    #     validation, container pulling, model downloading, data downloading
    #     and data uploading times.
    #
    #     **Potentially excludes model loading time,
    #     depending on the users implementation**.
    #
    #     Includes input data loading time, processing time,
    #     output data writing time and
    #     **any delays from shared hardware issues**.
    #
    # One, both or neither will be set.

    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process, predictions=predictions)

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    if metrics["results"]:
        metrics["aggregates"] = {
            "my_metric": mean(result["my_metric"] for result in metrics["results"])
        }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key(job)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("head-ct-angiography",): process_interf0,
        ("head-mr-angiography",): process_interf1,
    }[interface_key]

    # Call the handler
    return handler(job)


def process_interf0(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results

    location_aneurysm_segmentation = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="aneurysm-segmentation",
    )

    # Secondly, read the results

    result_aneurysm_segmentation = load_image_file_as_array(
        location=location_aneurysm_segmentation,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth

    image_name_head_ct_angiography = get_image_name(
        values=job["inputs"],
        slug="head-ct-angiography",
    )

    return evaluation_function(result_aneurysm_segmentation, image_name_head_ct_angiography)


def process_interf1(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results

    location_aneurysm_segmentation = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="aneurysm-segmentation",
    )

    # Secondly, read the results

    result_aneurysm_segmentation = load_image_file_as_array(
        location=location_aneurysm_segmentation,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth

    image_name_head_mr_angiography = get_image_name(
        values=job["inputs"],
        slug="head-mr-angiography",
    )

    return evaluation_function(result_aneurysm_segmentation, image_name_head_mr_angiography)


def log_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    # Each interface has a unique key that is the set of socket slugs given as input
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
