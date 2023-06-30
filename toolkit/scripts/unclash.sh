#!/bin/bash
# pkill -f "jjwang.*[c]lash"
pid=$(ps aux | grep jjwang | grep '[c]lash' | grep -v 'unclash.sh' | awk '{print $2}')
# 检查 $pid 变量是否为空
if [ -z "$pid" ]; then
  # 如果 $pid 变量为空
  echo -e "\033[0;31m# clash is not running.\033[0m" 
  # exit 0
else
  # 如果 $pid 变量非空
  unset http_proxy https_proxy all_proxy
  git config --global --unset http.https://github.com.proxy
  git config --global --unset https.https://github.com.proxy
  kill -9 $pid
  if [ $? -eq 0 ]; then
    echo -e "\033[0;32m# succeed to kill clash.\033[0m" 
  else
    echo -e "\033[0;31m# fail to kill clash.\033[0m" 
  fi
fi

