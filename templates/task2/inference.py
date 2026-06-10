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

    input_head_ct_angiography = load_image_file_as_array(
        location=INPUT_PATH / "images/head-ct-angio",
    )

    # Process the inputs: any way you'd like, here we show-case torch
    _show_torch_cuda_info()

    # Example how to set torch to use the GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.nn.Linear(10, 1).to(device)
    input = torch.randn(1, 10).to(device)
    output = model(input)

    # Your model will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # For now, let us make bogus predictions

    output_aneurysm_segmentation = numpy.eye(4, 2)

    # Save your output

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/aneurysm-segmentation",
        array=output_aneurysm_segmentation,
    )

    return 0


def interf1_handler():
    # Read the input

    input_head_mr_angiography = load_image_file_as_array(
        location=INPUT_PATH / "images/head-mr-angio",
    )

    # Process the inputs: any way you'd like, here we show-case torch
    _show_torch_cuda_info()

    # Example how to set torch to use the GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.nn.Linear(10, 1).to(device)
    input = torch.randn(1, 10).to(device)
    output = model(input)

    # Your model will be extracted to the `model_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./model` directory is mounted here.
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Algorithm and upload it under Models.
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # For now, let us make bogus predictions

    output_aneurysm_segmentation = numpy.eye(4, 2)

    # Save your output

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/aneurysm-segmentation",
        array=output_aneurysm_segmentation,
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


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
