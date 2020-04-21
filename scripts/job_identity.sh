#!/bin/bash
source ~/tf_env/bin/activate

python3 train_net.py \
--batch_size=32 \
--continue_train=1 \
--lr=5e-5 \
--summary_intv=1000 \
--validation_intv=5000 \
--loss=identity \
--tf_record_dir=./tf_record_dir_100k
