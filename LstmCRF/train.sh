#!/bin/bash
source activate py35
nohup                   \
python ./main.py     \
--train True  \
--char_dim=300  \
--hidden_dim=512  \
--batch_size=32  \
--require_improve=3000  \
--max_epoch=50 \
>> lstm.log  2>&1 &