#!/bin/bash
source activate py35
nohup                   \
python ./main.py     \
--train True  \
--char_dim=300  \
--hidden_dim=512  \
--batch_size=64  \
--require_improve=100  \
--max_epoch=50 \
>> lstm.log  2>&1 &