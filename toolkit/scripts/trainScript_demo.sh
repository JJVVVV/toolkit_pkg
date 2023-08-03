#!/bin/bash

# nohup ./trainScript_demo.sh > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0,1

# ###################################parameters#########################################
dashboard="tensorboard"
dataset_name="QQP"

model_type="bert-base-uncased"
model_dir="/data/jjwang/pretrained/$model_type"
model_name="baseline"

fp16=False
test_in_epoch=False
seed=5
accumulate_step=1
batch_size=32
batch_size_infer=32
epochs=5
max_length_input=None
learning_rate='2e-5'
weight_decay=0.01
metric='Accuracy'
warmup_ratio=0.1

train_file_path="data/$dataset_name/train/$part.jsonl"
val_file_path="data/$dataset_name/validation/$part.jsonl"
test_file_path=None
# ###################################parameters#########################################


# 判断有无console目录, 没有则创建
log_file="console/$dataset_name-$model_type-$model_name-$epochs-$batch_size-$learning_rate-$seed.ansi.log"
log_dir=${log_file%/*}
if [ ! -d log_dir ]; then
  mkdir -p $log_dir

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  --nproc-per-node=$nproc_pre_node \
  ./train.py \
    --dataset_name $dataset_name \
    --model_type $model_type \
    --model_name $model_name \
    --train_file_path $train_file_path \
    --val_file_path $val_file_path \
    --test_file_path $test_file_path \
    --seed $seed \
    --learning_rate $learning_rate \
    --epochs $epochs \
    --batch_size $batch_size \
    --batch_size_infer $batch_size_infer \
    --warmup_ratio $warmup_ratio \
    --max_length_input $max_length_input \
    --metric $metric \
    --test_in_epoch $test_in_epoch \
    --accumulate_step $accumulate_step \
    --fp16 $fp16 \
    --weight_decay $weight_decay \
    --dashboard $dashboard \
    --text_type $text_type \
    --min_threshold $min_threshold \
    --part $part \
    --model_dir $model_dir \
    > $log_file 2>&1 &
