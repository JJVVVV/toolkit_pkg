# bash download.sh

local_models_dir="~/pretrained_models"
# local_models_dir="/data/jjwang/pretrained/"

model_name="Qwen/Qwen2.5-7B-Instruct"

HF_ENDPOINT="https://hf-mirror.com"
hf_token=""

HF_ENDPOINT=$HF_ENDPOINT huggingface-cli download $model_name --local-dir local_models_dir --token $hf_token --exclude "*.msgpack" --exclude "*.h5" --exclude "*.bin" --exclude "*.ot" --exclude "original/"