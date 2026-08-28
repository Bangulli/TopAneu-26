#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="topaneu-26-task1-evaluation"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test_evaluations/docker-input"
OUTPUT_DIR="${SCRIPT_DIR}/test_evaluations/docker-output"
mkdir -pv "$INPUT_DIR"
# temp ground-truth dir for docker test by copying the contents
# of test_evaluations/ground-truth to GT_DIR with a location_jsons subdir
# (for load_gt() from evaluate.py to work inside docker env)
GT_DIR="${SCRIPT_DIR}/test_evaluations/docker-ground-truth"
mkdir -pv "$GT_DIR/location_jsons"
cp -avr "${SCRIPT_DIR}/test_evaluations/ground-truth/"* "$GT_DIR/location_jsons/"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read
chmod -R -f o+rX "$INPUT_DIR" "$GT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "$OUTPUT_DIR"

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -m o+rwX "$OUTPUT_DIR"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

echo "=+= Doing a forward pass"
## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# '--gpus all'
#    enables access to any GPUs present
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
# '--volume ../ground_truth:/opt/ml/input/data/ground_truth:ro'
#   is added to provide access to the (optional) tarball-upload locally
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "$GT_DIR":/opt/ml/input/data/ground_truth:ro \
    $DOCKER_IMAGE_TAG

echo "=+= Wrote results to ${OUTPUT_DIR}"

echo "=+= Verifying metrics.json against expected output"
EXPECTED_METRICS="${SCRIPT_DIR}/test_evaluations/output/expected_metrics.json"
ACTUAL_METRICS="${OUTPUT_DIR}/metrics.json"

python3 "${SCRIPT_DIR}/compare_docker_metrics.py" "$ACTUAL_METRICS" "$EXPECTED_METRICS"

echo "=+= Save this image for uploading via ./do_save.sh"

echo "=+= Cleaning up temp dirs"
# Use the container itself to remove /output contents, to avoid
# host-side permission errors on files written by the container
docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "$OUTPUT_DIR":/output \
    --entrypoint /bin/sh \
    $DOCKER_IMAGE_TAG \
    -c "rm -rf /output/* || true"

rm -rv "$GT_DIR"
