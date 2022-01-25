#!/bin/bash
#SBATCH --nodes=2 --ntasks=2 --cpus-per-task=40 -p gpumedium --gres=gpu:a100:4,nvme:1000 -t 00:15:00
#SBATCH -A dac

# Preparation:
# git clone https://github.com/huggingface/transformers
# pip install --user datasets transformers   #  just to get dependencies

export SINGULARITYENV_PYTHONPATH=$(pwd P)/transformers/src/

# first commit with fast multi-node
#SING_IMAGE=/scratch/project_2001659/mvsjober/singularity/pytorch_custom_3957ed4.sif

# first commit with regression
#SING_IMAGE=/scratch/project_2001659/mvsjober/singularity/pytorch_custom_38ac9e6.sif

if [ -z $SING_IMAGE ]; then
    echo "You must define singularity image with SING_IMAGE"
    exit 1
fi

SING_FLAGS="-B /users:/users -B /projappl:/projappl -B /scratch:/scratch -B $TMPDIR:$TMPDIR"

export HF_DATASETS_CACHE=$TMPDIR/datasets/

RDZV_HOST=$(hostname)
RDZV_PORT=29400

set -x

singularity exec $SING_FLAGS $SING_IMAGE python3 -c "import torch; print(torch.__version__)"

srun singularity exec $SING_FLAGS $SING_IMAGE python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    transformers/examples/pytorch/language-modeling/run_clm.py \
    --config_name config_nice.json \
    --tokenizer_name gpt2 \
    --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
    --output_dir $TMPDIR/output --per_device_train_batch_size 1 --max_steps 200

date
