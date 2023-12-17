# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92


# model_dir=${1:-"meta-llama/Llama-2-7b-chat"}

use_device="0,1"
use_device_="8,9"

log_dir=mp8
rm -rf $log_dir

# use_device="0"
export ASCEND_RT_VISIBLE_DEVICES=$use_device

model_dir="/home/data/dataset/llama-65b/llama_ptq_ckpts_smooth_all_shift_mp2_13b"

src_len=${2:-1100}
dec_len=${3:-330}
quant_type=${4:-"a8w8"}

total_len=`expr ${src_len} + ${dec_len}`

python -m paddle.distributed.launch \
    --log_dir $log_dir \
    --devices $use_device \
    predictor.py \
    --model_name_or_path $model_dir \
    --dtype float16 \
    --src_length ${total_len} \
    --max_length ${dec_len} \
    --output_file "infer.json" \
    --mode "static" \
    --batch_size 10 \
    --block_size 64 \
    --block_attn \
    --inference_model  \
    --use_cachekv_int8 static

# python read_res.py --model_name_or_path ${model_dir}