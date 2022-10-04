#!/bin/bash

# source activate nerf_pl

DATA_DIR=/home/baihy/datasets/nerf_synthetic/nerf_synthetic/lego

python train_packing.py \
   --dataset_name blender \
   --root_dir $DATA_DIR \
   --N_samples 64 \
   --N_importance 5 --img_wh 800 800 \
   --num_epochs 16 --batch_size 4096 \
   --lr 2e-3 \
   --lr_scheduler poly \
   --coord_scope 3.0 \
   --warmup_step 5000\
   --sigma_init 30.0 \
   --weight_threashold 1e-5 \
   --exp_name lego \

