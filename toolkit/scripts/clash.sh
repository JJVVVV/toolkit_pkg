#!/bin/bash
# pkill -f "jjwang.*[c]lash"
pid=$(ps aux | grep jjwang | grep '[c]lash' | grep -v 'clash.sh' | awk '{print $2}')
# 检查 $pid 变量是否为空
if [ -z "$pid" ]; then
  # 如果 $pid 变量为空
  ~/clash/clash -f ~/clash/lab.yaml &
  # export https_proxy=http://127.0.0.1:7895
  # export http_proxy=http://127.0.0.1:7895
  # export all_proxy=socks5://127.0.0.1:7896
  export https_proxy=http://jjwang:idonotknow@127.0.0.1:7895
  export http_proxy=http://jjwang:idonotknow@127.0.0.1:7895
  export all_proxy=socks5://jjwang:idonotknow@127.0.0.1:7896
  git config --global http.https://github.com.proxy http://jjwang:idonotknow@127.0.0.1:7895
  git config --global https.https://github.com.proxy http://jjwang:idonotknow@127.0.0.1:7895
  if [ $? -eq 0 ]; then
    echo -e "\033[0;32m# succeed to start clash.\033[0m" 
  else
    echo -e "\033[0;31m# fail to start clash.\033[0m" 
    # exit 0
  fi
else
  echo -e "\033[0;31m# clash is already running.\033[0m" 
fi

# cp bashScripts/ ~/ -r
# echo $https_proxy

