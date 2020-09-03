#!/bin/bash
source activate py35
nohup                   \
python ./main.py     \
--train False  \
--char_dim=300  \
--hidden_dim=512  \
--batch_size=32  \
--require_improve=3000  \
--max_epoch=50 \
>> predict.log  2>&1 &