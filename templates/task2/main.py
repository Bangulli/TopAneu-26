"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import glob
import json
from pathlib import Path
from inference import infer_ct, infer_mr
import numpy
import SimpleITK
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("head-ct-angiography",): interf0_handler,
        ("head-mr-angiography",): interf1_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input

    input_head_ct_angiography = load_image_file(
        location=INPUT_PATH / "images/head-ct-angio",
    )

    output_aneurysm_segmentation = infer_ct(input_head_ct_angiography)

    # Save your output

    write_image_file(
        location=OUTPUT_PATH / "images/aneurysm-segmentation",
        image=output_aneurysm_segmentation,
    )

    return 0


def interf1_handler():
    # Read the input

    input_head_mr_angiography = load_image_file(
        location=INPUT_PATH / "images/head-mr-angio",
    )

    output_aneurysm_segmentation = infer_mr(input_head_mr_angiography)

    # Save your output

    write_image_file(
        location=OUTPUT_PATH / "images/aneurysm-segmentation",
        image=output_aneurysm_segmentation,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return result


def write_image_file(*, location, image):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())
