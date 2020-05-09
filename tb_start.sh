#!/usr/bin/env bash

set -x
set -e

tb_start(){
    jobname="3ddl.${USER}.tensorboard"
    queuename=gpu_big

    if [ -n "$2" ]; then
   	 port=$1
   	 logdir=$2
    else
   	 port=32441
   	 logdir=$1
    fi

    screen -dmS tensorboard.$port | \
   	 srun -p $queuename -N 1 -J $jobname bash -c "module load python/tensorflow-1.14 && tensorboard --port $port --logdir $logdir"
    while [ -z "$(squeue -n $jobname -o %R -h)" ]; do sleep .5; done
    echo $(squeue -n $jobname -o %R -h):$port $logdir
}


tb_start $1