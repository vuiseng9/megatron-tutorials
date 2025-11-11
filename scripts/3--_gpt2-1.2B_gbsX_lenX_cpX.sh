#!/bin/bash

GBS=${1:?"Error: No input argument, pls specify GBS"}
SL=${2:?"Error: No input argument, pls specify sequence length"}
CP=${3:?"Error: No input argument, pls specify CP degree"}
RA=${4:-0} 
COMM=${5:-"p2p"}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=$CP
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

if [ "$RA" -eq 0 ]; then
    ratag=""
    TRAINING_ARGS=""
elif [ "$RA" -eq 1 ]; then
    ratag="_recompute_act"
    TRAINING_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
else
    echo "Error: last argument must be 0 or 1 (recompute activation)"
    exit
fi

TS=$(date +"%y%m%d_%H%M%S")
RUN_ID=${TS}___gpt2-1.2B-${NUM_NODES}x${GPUS_PER_NODE}gpu_gbs${GBS}_len${SL}_cp${CP}_${COMM}${ratag}
OUTDIR="./outdir/${RUN_ID}"
TENSORBOARD_LOGS_PATH=$OUTDIR/tb #<Specify path>
mkdir -p $TENSORBOARD_LOGS_PATH

MLMROOT="../../"
export PYTHONPATH=$MLMROOT:$PYTHONPATH

DSDIR="./owt-ds"
VOCAB_FILE=$DSDIR/gpt2-vocab.json
MERGE_FILE=$DSDIR/gpt2-merges.txt
DATA_PATH=$DSDIR/openwebtext-10k_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# gpt2 1.2B params
GPT_MODEL_ARGS=(
    --hidden-size 1536
    --num-attention-heads 16
    --num-layers 40 
    --seq-length $SL
    --max-position-embeddings $SL
    --attention-backend flash # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS+=(
    --micro-batch-size $GBS 
    --global-batch-size $GBS 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
    --context-parallel-size $CP
    --cp-comm-type $COMM
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --train-iters 100
    --log-interval 1
    --log-throughput
    --save-interval 0 
    --eval-interval 25 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $MLMROOT/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"

printf "Executing...\n${cmd}\n"
eval $cmd 2>&1 | tee $OUTDIR/logs.txt
