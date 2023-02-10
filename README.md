# Novel View Synthesis with Diffusion Models

Unofficial PyTorch Implementation of [Novel View Synthesis with Diffusion Models](https://3d-diffusion.github.io/).

## Changes:

**Distributed version:**
- [x] Running PyTorch Lightning version
- [ ] PyTorch DDP support for large scale training 

## Training:

### PyTorch DDP
Yet to test, copied from the doc:
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py

```

### PyTorch Lightning
```
python train.py --train_data path_to_data --transfer path_to_ckpt
```

## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download `chairs_train.zip` and `cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

We include pickle file that contains available view-png files per object. 

## TODO:
1. Assessing the behavior of the loss along training
2. Testing on a cluster of multiple gpus

## Pre-trained Model Weights:

[Google Drive](https://drive.google.com/file/d/1GarX4DA2FNPHeAUbzSkV1RuJC0Ci-SE5/view?usp=sharing)

We trained SRN Car dataset for 101K steps for 120 hours. We have tested using 8 x RTX3090 with batch size of 128 and image size of 64 x 64. Due to the memory constraints, we were not able to test the original authors' configuration of image size 128 x 128.
