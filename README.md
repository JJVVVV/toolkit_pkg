# toolkit

## Install
```shell
git clone https://github.com/JJVVVV/toolkit.git
cd path/to/toolkit
pip install --editable .
```

## Update
```shell
git pull origin
```


## Todo list
1. 使用额外的 optimizer 和 scheduler

    相关doc:

    1. https://deepspeed.readthedocs.io/en/latest/schedulers.html
    2. https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/deepspeed.py#L259
    
2. 保存多个最优模型（当前只能保存一个最优的）



## 常见问题：
1. 安装deepspeed时报错：FileNotFoundError: [Errno 2] No such file or directory: ':/usr/local/cuda/bin/nvcc'\
检查CUDA_HOME=/usr/local/cuda是否配置正确。https://github.com/deepspeedai/DeepSpeed/issues/5511
2. 