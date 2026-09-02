"""
The following is a simple example evaluation method.

You can run the evaluation locally outside of Docker environment.
Outside of Docker, by default, main.py looks for files to evaluate in the current directory:
in ./predictions/ and ./ground-truth/
You can override this with any folder containing `ground-truth/` and `predictions/` sub-folders.
The naming of gt and pred files can be arbitrary as long as their filenames are sorted in the same order.

python3 main.py --base_path <parent_dir_of_gt_pred_subdirs>

When run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

(for organizers) Test for Docker environment
  ./do_test_run.sh

(for organizers) To save the container and prep it for upload to Grand-Challenge.org you can call:
  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json
import logging
from pathlib import Path
from pprint import pformat

from evaluate import evaluation_aggregation, evaluation_average, evaluation_function
from helpers import is_docker, run_prediction_processing, setup_logger, tree

logger = logging.getLogger("evaluate")

# supported extensions for evaluation
EXTENSIONS = ("*.json",)

EXEC_IN_DOCKER = is_docker()

if EXEC_IN_DOCKER:
    # input dir with the predictions.json file for Docker env
    # need to define here before the process interface functions
    INPUT_DIRECTORY = Path("/input")


def main():
    metrics = {}

    if EXEC_IN_DOCKER:
        setup_logger(
            # Optionally: change this to the more verbose DEBUG
            level=logging.INFO,
        )

        log_inputs()

        # fetch the predictions through predictions.json
        predictions = read_predictions()

        # Use concurrent workers to process the predictions more efficiently
        metrics["results"] = run_prediction_processing(
            fn=process, predictions=predictions
        )
    else:
        # When not in docker environment, put gt and pred in these folders
        # Make sure the gt/pred appear in the same sorted order
        pred_dir = BASE_PATH / "predictions"
        gt_dir = BASE_PATH / "ground-truth"

        predictions = rglob_files(pred_dir, EXTENSIONS)
        gts = rglob_files(gt_dir, EXTENSIONS)

        print(f"predictions = {predictions}")
        print(f"gts = {gts}")

        assert len(predictions) > 0, "no prediction files"
        assert len(gts) > 0, "no ground truth"

        # early abort if num pred != gt files
        assert len(gts) == len(predictions), "unequal gt & pred"

        metrics["results"] = []

        for i, gt_path in enumerate(gts):
            print(f"i = {i}")
            print(f"gt_path = {gt_path}")

            pred_path = predictions[i]

            print(f"pred_path = {pred_path}")

            detected_aneurysm_locations = load_json_file(
                path=pred_path,
            )

            result = evaluation_function(
                detected_aneurysm_locations,
                gt_path,
                execute_in_docker=False,
            )
            metrics["results"].append(result)

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

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    if metrics["results"]:
        metrics["aggregates_per_location"] = evaluation_aggregation(metrics["results"])
        metrics["aggregates_avg"] = evaluation_average(
            metrics["aggregates_per_location"]
        )

        # document the number of test images
        metrics["n_cases"] = len(metrics["results"])

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key(job)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("head-ct-angiography",): process_interf_ct,
        ("head-mr-angiography",): process_interf_mr,
    }[interface_key]

    # Call the handler
    return handler(job)


def process_interf_ct(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the path of the results

    path_detected_aneurysm_locations = get_pred_file_path(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-aneurysm-locations",
    )

    # Secondly, read the results

    detected_aneurysm_locations = load_json_file(
        path=path_detected_aneurysm_locations,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth

    image_name_head_ct_angiography = get_image_name(
        values=job["inputs"],
        slug="head-ct-angiography",
    )

    return evaluation_function(
        detected_aneurysm_locations,
        image_name_head_ct_angiography,
    )


def process_interf_mr(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the path of the results

    path_detected_aneurysm_locations = get_pred_file_path(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-aneurysm-locations",
    )

    # Secondly, read the results

    detected_aneurysm_locations = load_json_file(
        path=path_detected_aneurysm_locations,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth

    image_name_head_mr_angiography = get_image_name(
        values=job["inputs"],
        slug="head-mr-angiography",
    )

    return evaluation_function(
        detected_aneurysm_locations,
        image_name_head_mr_angiography,
    )


def log_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


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


def get_pred_file_path(*, job_pk, values, slug):
    # Where a job's output file will be in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, path) -> list[int]:
    """
    compatible with both json of a naked list or nested json with `locations` key

    Supports either:
      [1, 2, 3]
    or:
      {"locations": [1, 2, 3]}

    Returns the aneu locations list.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "locations" in data:
        return data["locations"]

    raise ValueError("Expected a list or an object containing a 'locations' key")


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def rglob_files(folder, extensions):
    """for non-docker local evaluation"""
    return sorted(
        [
            f
            for ext in extensions
            for f in folder.rglob(ext)
            if f.name != "predictions.json" and f.name != "inputs.json" and f.is_file()
        ]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).parent,
        help=(
            "(Optional) Specify the base directory containing "
            "the predictions/ and ground-truth/ directories "
            "for local non-Docker evaluation. "
            "Defaults to the directory containing this script."
        ),
    )

    args = parser.parse_args()

    if EXEC_IN_DOCKER:
        BASE_PATH = Path("/")
    else:
        BASE_PATH = args.base_path

    print(f"BASE_PATH = {BASE_PATH}")

    # output dir for the metrics.json file
    OUTPUT_DIRECTORY = BASE_PATH / "output"

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    raise SystemExit(main())
