#!/bin/bash

# Runs the "175B" parameter model
SP=${1:-0}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=172.16.0.72
MASTER_PORT=29500
NUM_NODES=2
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

RUNDIR="./sandbox_run"
CHECKPOINT_PATH=$RUNDIR/ckpt #<Specify path>
TENSORBOARD_LOGS_PATH=$RUNDIR/tb #<Specify path>
mkdir -p $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH

MLMROOT="../../"
export PYTHONPATH=$MLMROOT:$PYTHONPATH

DSDIR="./owt-ds"
VOCAB_FILE=$DSDIR/gpt2-vocab.json
MERGE_FILE=$DSDIR/gpt2-merges.txt
DATA_PATH=$DSDIR/openwebtext-10k_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES
    --rdzv_backend c10d
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_id multnode_mlm_tp8
)

GPT_MODEL_ARGS=(
    --hidden-size 3072
    --num-attention-heads 32
    --num-layers 72 
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

MBS=8
GBS=$(($MBS*$NUM_NODES))
    # --rampup-batch-size 16 16 5859375 
TRAINING_ARGS=(
    --micro-batch-size $MBS 
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

TP=$GPUS_PER_NODE
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size 1 
)

if [ "$SP" -eq 0 ]; then
    sp="no-sp"
    :
elif [ "$SP" -eq 1 ]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
    sp="wt-sp"
else
    echo "Error: last argument must be 0 (no SP) or 1 (with SP)"
    exit
fi

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 50 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-entity vchua
    --wandb-project mlm-sandbox
    --wandb-exp-name $(date +"%y%m%d_%H%M%S")_weak_tp${TP}_${sp}_${GPUS_PER_NODE}gpus_gbs${GBS}_gpt2-8.3B
)

echo "Executing..."
echo "torchrun ${DISTRIBUTED_ARGS[@]} $MLMROOT/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"

torchrun ${DISTRIBUTED_ARGS[@]} $MLMROOT/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
