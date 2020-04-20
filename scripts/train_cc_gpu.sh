#!/bin/bash
#SBATCH --account=def-kyi
# SBATCH --output=/home/abojja9/scratch/SiamesePersonReID/outputs/augment_identity_%x-%j.out

TODO_DIR="./jobs/todo"
DONE_DIR="./jobs/done"

# Create DONE directory if it does not exist
if [ ! -d $DONE_DIR ]; then
    mkdir -p $DONE_DIR
fi

# Setup environment
source $HOME/.bashrc
workon tensorflow-gpu

# Run job
FAIL=0
GPU_ID=0
for CUR_JOB_SH in "$@" ; do
    # Run a job
    if [[ $CUR_JOB_SH != "" ]] ; then
        # # Move it to the done list
        # mv ${QUEUE_DIR}/${CUR_JOB_SH} ${DONE_DIR}/${CUR_JOB_SH}
        # Report which job I am running
        echo "Running ${CUR_JOB_SH}"
        # Run Script at background
        OMP_NUM_THREAD=16 CUDA_VISIBLE_DEVICES=$GPU_ID bash ${DONE_DIR}/${CUR_JOB_SH} &
        # echo $CUR_JOB_SH
        # Sleep 5 sec
        sleep 5
    fi

    let "GPU_ID=GPU_ID+1"
done

for job in `jobs -p`
do
    echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
    echo "YAY!"
else
    echo "FAIL! ($FAIL)"
fi
