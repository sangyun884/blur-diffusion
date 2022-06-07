#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

python eval_x0hat.py \
--name test \
--noise_schedule linear \
--gpu 0 \
--bsize 4 \
--sig 0.4 \
--sig_min 0 \
--sig_max 0.15 \
--f_type quartic \
--res 64 \
--nc 3 \
--loss_type eps_simple \
--ckpt /home/ubuntu/code/sangyoon/forward-blur-new/experiments/lsun-simplified-64-sigmax0.15-quartic/model_450001.ckpt
