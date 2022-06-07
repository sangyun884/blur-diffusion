export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python main.py \
--name test \
--dataset lsun-bedroom \
--use_ema \
--ema_decay 0.9999 \
--noise_schedule linear \
--f_type quartic \
--gpu 0 \
--bsize 16 \
--sig 0.4 \
--sig_min 0 \
--sig_max 0.15 \
--fid_bsize 16 \
--loss_type eps_simple \
--lr 0.00005 \