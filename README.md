# Test case for reproducing multi-node slowdown regression issue in PyTorch

## Preparation

Create singularity container according to recipe
[pytorch_git_custom.def](pytorch_git_custom.def).


```
git clone https://github.com/huggingface/transformers
pip install --user datasets transformers   #  just to get dependencies
```

## Running

```
SING_IMAGE=/path/to/singularity_image/pytorch_custom_COMMIT.sif sbatch pytorch_clm_test.sh
```
