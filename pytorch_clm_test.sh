#!/bin/bash
#SBATCH --nodes=2 --ntasks=2 --cpus-per-task=64 -p gpumedium --gres=gpu:a100:4,nvme:1000 -t 00:15:00
#SBATCH -A dac

module purge
module load pytorch

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

date
hostname

export HF_DATASETS_CACHE=/scratch/dac/mvsjober/hf/datasets/
export HF_METRICS_CACHE=/scratch/dac/mvsjober/hf/metrics/
export HF_MODULES_CACHE=/scratch/dac/mvsjober/hf/modules/

RDZV_HOST=$(hostname)
RDZV_PORT=29400

SING_IMAGE=/scratch/project_2001659/mvsjober/singularity/pytorch_a50a389.sif
#SING_IMAGE=/scratch/project_2001659/mvsjober/singularity/pytorch_38ac9e6.sif

SING_FLAGS="-B /users:/users -B /projappl:/projappl -B /scratch:/scratch"
SING_FLAGS="$SING_FLAGS -B $TMPDIR:$TMPDIR -B $LOCAL_SCRATCH:$LOCAL_SCRATCH"

rm -rf /scratch/dac/mvsjober/hf/output/*

set -x

singularity exec $SING_IMAGE $SING_FLAGS python3 -c "import torch; print(torch.__version__)"

srun singularity exec $SING_IMAGE $SING_FLAGS python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    transformers/examples/pytorch/language-modeling/run_clm.py \
    --config_name config_nice.json \
    --tokenizer_name gpt2 \
    --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
    --output_dir /scratch/dac/mvsjober/hf/output --per_device_train_batch_size 1 --max_steps 200"

date
