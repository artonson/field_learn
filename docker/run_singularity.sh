#!/bin/bash

set -e
set -x

# example launch string:
# ./run_singularity.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>
#   server_data_dir:        the data directory where the dataset sample resides
#   server_logs_dir:        the directory where the output logs are supposed to be written
#   gpu:                    comma-separated list of gpus

module load apps/singularity-3.2.0

if [[ $# -lt 2 ]]
then
    echo "run_singularity.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>"
    exit 1
fi

while getopts "d:l:g:" opt
do
    case ${opt} in
        d) HOST_DATA_DIR=$OPTARG;;
        l) HOST_LOG_DIR=$OPTARG;;
        g) GPU_ENV=$OPTARG;;
        *) echo "No reasonable options found!";;
    esac
done

if [[ ! -d ${HOST_DATA_DIR} ]]; then
    echo "server_data_dir is not set or not a directory";
    exit 1
fi
if [[ ! -d ${HOST_LOG_DIR} ]]; then
    echo "server_logs_dir is not set or not a directory";
    exit 1
fi


# Copy the exact same code used for building the container
# SIMAGES_DIR=/gpfs/gpfs0/3ddl/field_learn/singularity-images
# IMAGE_NAME="mariataktasheva/fieldlearn"
SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/vectran"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

# Build the container if it does not exist
[[ -f ${SIMAGE_FILENAME} ]] || ./build_singularity.sh


HOST_CODE_DIR=$(realpath $(dirname `realpath $0`)/../..)     # dirname of THIS file
CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_LOG_DIR="/logs"

if [[ -z "${GPU_ENV}" ]] ; then
    # set all GPUs as visible in the docker
    num_gpus=`nvidia-smi -L | wc -l`
    GPU_ENV=`seq -s, 0 $((num_gpus-1))`
fi

echo "******* LAUNCHING CONTAINER ${SIMAGE_FILENAME} *******"
echo "      Pushing you to ${CONT_CODE_DIR} directory"
echo "      Data is at ${CONT_DATA_DIR}"
echo "      Writable logs are at ${CONT_LOG_DIR}"
echo "      Environment: PYTHONPATH=${CONT_CODE_DIR}"
echo "      Environment: CUDA_VISIBLE_DEVICES=${GPU_ENV}"

CUDA_VISIBLE_DEVICES=${GPU_ENV} \
PYTHONPATH=${CONT_CODE_DIR} \
    singularity shell \
        --nv \
        --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
        --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
        --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
        --bind $PWD:/run/user \
        --workdir ${CONT_CODE_DIR} \
        ${SIMAGE_FILENAME}

