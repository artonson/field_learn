#!/bin/bash

set -e
set -x

if [[ $# -lt 2 ]]
then
    echo "run_singularity.sh -c <config_path>"
    exit 1
fi

#Get all the required options and set the necessary variables
while getopts "c:" opt
do
    case ${opt} in
        c) CONFIG_PATH=$OPTARG;;
        *) echo "No reasonable options found!";;
    esac
done

if [[ ! -f ${CONFIG_PATH} ]]; then
    echo "config path is not set or not a file";
    exit 1
fi

# make a model tag from config name
TAG=$(basename -- "$CONFIG_PATH")
EXTENTION="${TAG##*.}"
TAG="${TAG%.*}"

module load apps/singularity-3.2.0

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/vectran"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

HOST_CODE_DIR="/gpfs/data/home/m.taktasheva/github" 
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/"
HOST_LOG_DIR="/gpfs/data/home/m.taktasheva/tmp/docker/logs"
CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_LOG_DIR="/logs"

TRAIN_SCRIPT="${CONT_CODE_DIR}/field_learn/scripts/train_polyvector_field_regression.py"

num_gpus=`nvidia-smi -L | wc -l`
GPU_ENV=`seq -s, 0 $((num_gpus-1))`

echo "******* LAUNCHING CONTAINER ${SIMAGE_FILENAME} *******"
echo "      Pushing you to ${CONT_CODE_DIR} directory"
echo "      Data is at ${CONT_DATA_DIR}"
echo "      Writable logs are at ${CONT_LOG_DIR}"
echo "      Environment: PYTHONPATH=${CONT_CODE_DIR}"
echo "      Environment: CUDA_VISIBLE_DEVICES=${GPU_ENV}"
echo "      Model config: ${CONFIG_PATH}"
echo "      Model tag: ${TAG}"


CUDA_VISIBLE_DEVICES=${GPU_ENV} \
PYTHONPATH=${CONT_CODE_DIR} \
    singularity exec \
        --nv \
        --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
        --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
        --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
        --bind $PWD:/run/user \
        --workdir ${CONT_CODE_DIR} \
        ${SIMAGE_FILENAME} \
        bash -c "python ${TRAIN_SCRIPT} \\
                --model-tag ${TAG} \\
                --config ${CONFIG_PATH}"

