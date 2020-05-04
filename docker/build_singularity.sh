#!/bin/bash

set -e
set -x

# example launch string:
# ./build_singularity.sh 

SIMAGES_DIR=/gpfs/gpfs0/3ddl/field_learn/singularity-images

[[ -d ${SIMAGES_DIR} ]] || mkdir ${SIMAGES_DIR}

IMAGE_NAME="mariataktasheva/fieldlearn"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

echo "******* PULLING IMAGE FROM DOCKER HUB AND BUILDING SINGULARITY IMAGE *******"
singularity pull "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
