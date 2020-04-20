#!/bin/bash

# Excute by saying ./queue_train_gpu.sh 10 1 "nodep"

TODO_DIR="./jobs/todo"
DONE_DIR="./jobs/done"

# Each job is run this much
RUN_TIMES=$1
shift

# Total jobs to queue
TOTAL_JOBS=$1
shift

# Depends key
DEPENDS_KEY=$1
shift

HOSTNAME=$(hostname)
# Set options depending on the cluster
if [[ "$HOSTNAME" == "gra"* ]] ; then
    NUM_CPU="16"
    NUM_GPU="1"
    MEM_SIZE="120000M"
elif [[ "$HOSTNAME" == "cedar"* ]] ; then
    NUM_CPU="16"
    NUM_GPU="1"
    MEM_SIZE="100000M"
elif [[ "$HOSTNAME" == "beluga"* ]] ; then
    NUM_CPU="32"
    NUM_GPU="1"
    MEM_SIZE="180000M"
fi

# Set time limit
TIME_LIMIT="0-03:00"

# Find current jobs that I should depend on
DUMP_JOBS=($(squeue -u abojja9 | grep "${DEPENDS_KEY}" | awk '{print $1}'))
function join { local IFS="$1"; shift; echo "$*"; }

# Run the job
for JOB_NUM in $(seq 1 $TOTAL_JOBS) ; do
    # Create the job array by grabbing jobs
    JOB_SCRIPTS=()
    for GPU_ID in $(seq 1 $NUM_GPU) ; do
	
	   # Grab a job
	CUR_JOB_SH=$(ls -rt $TODO_DIR/run_segnet.sh | head -1)
	if [[ "$CUR_JOB_SH" != "" ]] ; then
	    CUR_JOB_SH=$(echo $CUR_JOB_SH | xargs basename)
	    # Move that job to queue
	    mv ${TODO_DIR}/${CUR_JOB_SH} ${DONE_DIR}/${CUR_JOB_SH}
	    # Add to the list
	    JOB_SCRIPTS+=($CUR_JOB_SH)
	fi
    done
    # Dispatch jobs
    if [[ $#JOB_SCRIPTS != 0 ]] ; then
	# Initial dependency (e.g. any dump jobs)
	JOB_ID=""
	DEPENDS_ON=""
	JOB_ID=$(join : "${DUMP_JOBS[@]}")
	if [[ "$JOB_ID" != "" ]] ; then
	    DEPENDS_ON="--dependency=afterany:$JOB_ID"
	fi
	# Other jobs
	for RUN_NUM in $(seq 1 $RUN_TIMES) ; do 
	    SLURM_OUT=$(sbatch "$@" --cpus-per-task=$NUM_CPU \
			       --gres=gpu:$NUM_GPU --mem=$MEM_SIZE \
			       --time=$TIME_LIMIT $DEPENDS_ON \
			       ./train_cc_gpu.sh "${JOB_SCRIPTS[@]}")
	    echo $SLURM_OUT
	    JOB_ID=$(echo $SLURM_OUT | awk '{print $4}')
	    DEPENDS_ON="--dependency=afterany:$JOB_ID"
	done
    else
	# We are done
	echo "No more jobs to queue!"
	exit 0
    fi
done
