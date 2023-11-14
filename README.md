# SBCFormer-Pytorch
## paper:SBCFormer: Lightweight Network Capable of Full-size ImageNet Classification at 1 FPS on Single Board Computers(https://arxiv.org/abs/2311.03747)

## Precautions
Note: Before training the classification model, you need to enter the __train_gpu.py__ file to modify the __data_root__ and __batchsize__ of your own data set.The code turns on automatic mixed precision by default. If your GPU(s) does not support automatic mixed precision (you can use __torch.cuda.is_bf16_supported()__ to check), you need to modify it in the __util.engine.py__ file.

## Train multi-gpus
Run command: torchrun --nproc_per_node=8 train_gpu.py

## Train with specify specific GPUs
Run command: CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 train_gpu.py

## Train single-gpu
Run command: python train_gpu.py
