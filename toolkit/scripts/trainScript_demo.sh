#!/bin/bash

# nohup ./trainScript_bert_QQP.sh > /dev/null 2>&1 &
# pkill -s SIGKILL -pgn 3697090

# 定义一个数组，存放种子
# while kill -0 $PID 2>/dev/null; do sleep 1; done


# # QQP
# 25571765677776765
seeds=(5 17 25 57 65 67 76 77)
# seeds=(5)
CUDA_VISIBLE_DEVICES=0/

# ###################################parameters#########################################
dashboard="wandb"
dataset_name="QQP"
part="all"
text_type='ORI'

min_threshold=0.7

model_type="bert-base-uncased"
model_dir="/data/jjwang/pretrained/$model_type"
# model_type="bert-large-uncased"
# model_type='hfl/chinese-bert-wwm-ext'

model_name="baseline"
# model_name="noise_$min_threshold"

fp16=False
test_in_epoch=True

accumulate_step=1
batch_size=32
batch_size_infer=32
epochs=5
max_length_input=None
learning_rate='2e-5'
weight_decay=0.1
metric='Accuracy'

train_file_path="data/$dataset_name/train/$part.jsonl"
val_file_path="data/$dataset_name/validation/$part.jsonl"
test_file_path=None

warmup_ratio=0.01
# ###################################parameters#########################################

# 定义一个数组，存放可用cuda
# IFS=',' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
IFS='/' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
# 计算每个每个任务可用cuda数量
IFS=',' nproc_pre_node=(${cudas[0]}) IFS=' '
nproc_pre_node=${#nproc_pre_node[@]}
# 定义一个变量，表示最大并行数
parallel=${#cudas[@]}
# 定义一个数组，存放当前运行的进程号
pids=()
# 定义一个字典, 记录PID运行在哪个CUDA设备上
declare -A pid_cuda


# 遍历所有的种子
for seed in ${seeds[@]}
do
  # 判断有无console目录, 没有则创建
  log_file="console/$dataset_name-$text_type-$model_type-$model_name-$epochs-$batch_size-$learning_rate-$seed.ansi.log"
  log_dir=${log_file%/*}
  if [ ! -d log_dir ]; then
    mkdir -p $log_dir
  fi
  # 如果当前运行的进程数达到最大并行数，就等待任意一个进程结束: 从数组pids中删除结束进程的PID, 释放一个CUDA
  if [ ${#pids[@]} -eq $parallel ]; then
    wait -n ${pids[@]}
    # 删除已经结束的进程号, 释放一个可用的cuda
    for pid in ${pids[@]}
    do
      if ! ps -p $pid > /dev/null ; then
        # echo $pid
        finishedPID=$pid
        break
      fi
    done
    echo "finishPID: $finishedPID"
    pids=(${pids[@]/$finishedPID/})
    cudas+=(${pid_cuda[$finishedPID]})
    echo "freeCUDA: ${pid_cuda[$finishedPID]}"
    unset pid_cuda[$finishedPID]
    echo "runningProcesses: ${pids[@]}"
    echo "avaliableCUDAs: ${cudas[@]}"
    echo
  fi
  # 启动一个新训练任务: 使用一个可用的cuda,并把它的PID添加到数组pids中
  cuda=${cudas[0]}
  unset cudas[0]
  cudas=(${cudas[@]})
  # sed -i "s/seed=.*/seed=$seed/" ./trainScript.sh
  # sed -i "s/CUDA_VISIBLE_DEVICES=[0-9|,]*/CUDA_VISIBLE_DEVICES=$cuda/" ./trainScript.sh
  # ./trainScript.sh > "console//seed-$seed.log" 2>&1 &
  # ###################################训练程序#########################################
  # HUGGINGFACE_HUB_CACHE="/data/jjwang/.cache/huggingface/hub/" TRANSFORMERS_CACHE="/data/jjwang/.cache/huggingface/hub/" \
  # TORCH_DISTRIBUTED_DEBUG=INFO \
  CUDA_VISIBLE_DEVICES=$cuda \
  torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=$nproc_pre_node \
    ./train_trainer.py \
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
    # --fp16 \
    # --batch_size $(($batch_size/$nproc_pre_node)) \
    # --pretrained_model_path $pretrained_model_path \
    # --early_stop \
    # --early_stop_metric $early_stop_metric \
    # --continue_train_more_patience \
    # --max_length $max_length \
    # --continue_train_more_epochs \
    # --continue_train_more_patience \
          # --do_lower_case \
  # ###################################训练程序#########################################
  newPID=$!
  pids+=($newPID)
  pid_cuda[$newPID]=$cuda
  echo "newPID: $newPID"
  echo "useCUDA: ${pid_cuda[$newPID]}"
  echo "runningProcesses: ${pids[@]}"
  echo "avaliableCUDAs: ${cudas[@]}"
  echo

  # while [ ! -f "console/seed-$seed.log" ]; do
  #   echo "waiting trainScript.sh to run in the background."
  #   sleep 1
  # done

done
# 等待所有剩余的进程结束
wait ${pids[@]}