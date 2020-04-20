#!/bin/bash
# source activate LSD-seg_v1
# python main.py --dataroot '/home/abojja/stn_gan/test-time-domain-adaptation/synthia-seg/data' --checkpoint "/data/scratch/abojja/sept_2019/checkpoint"  --network='segnet' --continue_train=0  --phase=train --height=320 --width=640 --batch_size=8 
source ~/tf_env/bin/activate

python3 train_net.py \
--batch_size=32 \
--continue_train=1 \
--lr=3e-3 \
--summary_intv=1000 \
--validation_intv=5000 \
--loss=combined \
--tf_record_dir=./tf_record_dir_100k
