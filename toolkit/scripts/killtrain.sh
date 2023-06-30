#!/bin/bash

# # 结束Auto进程及其子进程
# killall -KILL -u jjwang -v trainAuto.sh
# if [ $? -eq 0 ]; then
#   echo -e "\033[0;32m# succeed to kill trainAuto.\033[0m" 
# else
#   echo -e "\033[0;31m# fail to kill trainAuto.\033[0m" 
# fi

# 结束GPU上的进程
# pids=($(gpustat -cpu | grep 'jjwang' | sed 's/.*jjwang:python\/\([0-9]*\).*/\1/'))
pids=($(ps aux | grep jjwang | grep ./[t]rain | grep -v 'killtrain.sh' | awk '{print $2}'))
for pid in ${pids[@]}
do
  # echo $(($pid))
  kill -9 $(($pid))
  if [ $? -eq 0 ]; then
    echo -e "\033[0;32m# succeed to kill $pid.\033[0m" 
  else
    echo -e "\033[0;31m# fail to kill $pid.\033[0m" 
  fi
done
exit 0