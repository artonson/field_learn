#!/bin/bash

set -e
set -x

# example launch string:
# ./build_singularity.sh [-p]

module load apps/singularity-3.2.0

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images

[[ -d ${SIMAGES_DIR} ]] || mkdir ${SIMAGES_DIR}

IMAGE_NAME="artonson/vectran"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

echo "******* PULLING IMAGE FROM DOCKER HUB AND BUILDING SINGULARITY IMAGE *******"
singularity pull "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
