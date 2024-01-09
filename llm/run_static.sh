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

# export FLAGS_call_stack_level=2
# export GLOG_logtostderr=true
# export GLOG_v=0

# export FLAGS_control_flow_use_new_executor=1
# export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.98
# export FLAGS_allocator_strategy=auto_growth
# export FLAGS_eager_delete_tensor_gb=1.0

# 调试打开
# export FLAGS_npu_blocking_run=true
# export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
# export ATB_PROFILING_ENABLE=1
 
# CANN日志
# export ASCEND_GLOBAL_EVENT_ENABLE=0
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1

# 算子库日志
# export ASDOPS_LOG_LEVEL=INFO
# # export ASDOPS_LOG_TO_STDOUT=1
# export ASDOPS_LOG_TO_FILE=1
# export ASDOPS_LOG_TO_FILE_FLUSH=1
# rm -f /root/atb/log/atb_*
# cp -r /root/atb/log/ ./atb_log

# model_dir=${1:-"meta-llama/Llama-2-7b-chat"}

# quant下届配适
export ASDOPS_QUANT_MIN_NEG_127=1

# 内存算法
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

#EVENT消减
export FLAGS_use_stream_safe_cuda_allocator=0

# 使能Lccl
export HCCL_BUFFSIZE=120

# 确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=1

use_device="0,1,2,3,4,5,6,7"
use_device_="8,9,10,11,12,13,14,15"

log_dir=mp8
rm -rf $log_dir

export ASCEND_RT_VISIBLE_DEVICES=$use_device

model_dir="/home/data/dataset/llama-65b/llama_ptq_ckpts_smooth_all_shift_mp2_13b"

src_len=${2:-1100}
dec_len=${3:-330}
quant_type=${4:-"a8w8"}

total_len=`expr ${src_len} + ${dec_len}`

# python -m paddle.distributed.launch \
#     --log_dir $log_dir \
#     --devices $use_device \
#     predictor.py \
#     --model_name_or_path $model_dir \
#     --dtype float16 \
#     --src_length ${total_len} \
#     --max_length ${dec_len} \
#     --output_file "infer.json" \
#     --mode "static" \
#     --batch_size 10 \
#     --block_size 64 \
#     --block_attn \
#     --inference_model  \
#     --use_cachekv_int8 static

# python read_res.py --model_name_or_path ${model_dir}

declare -A map
# 1、通过npu-smi info，确认每个卡的Bus-Id
# 2、通过lspci -vvv -s <Bus-Id>,确认每个卡numa node 亲和性
map["0"]="0"
map["1"]="0"
map["2"]="0"
map["3"]="0"
map["4"]="1"
map["5"]="1"
map["6"]="1"
map["7"]="1"

RANK_ID_START=0
WORLD_SIZE=8

# bs_array=(28 34 35 52)
bs_array=(52)
length=${#bs_array[@]}
echo "Run $length"

for ((i=0; i<$length; i++)); do

    BATCH_NUM=${bs_array[$i]}
    TOTAL_LEN=${total_len}
    MAX_LEN=${dec_len}

    echo "BATCH_NUM: $BATCH_NUM, TOTAL_LEN: $TOTAL_LEN, MAX_LEN: $MAX_LEN"

    if test -d "$model_dir";
    then
        echo "Weight directory exists, runing......"
        for((RANK_ID=$RANK_ID_START;RANK_ID<$WORLD_SIZE;RANK_ID++));
        do
        bind=${map["$RANK_ID"]}
        echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
        numactl --cpunodebind=$bind --membind $bind python -m paddle.distributed.launch --master 127.0.0.1:49122 --rank $RANK_ID --devices $RANK_ID --nnodes $WORLD_SIZE python3 predictor.py --model_name_or_path $model_dir --src_length $TOTAL_LEN --max_length $MAX_LEN --batch_size $BATCH_NUM --block_size 128 --block_attn --top_p 0.0 --decode_strategy greedy_search --dtype "float16" --mode "static" --device "npu" --benchmark --use_cachekv_int8 static --inference_model 1 &> $RANK_ID.log &
    done
    fi
    # tail -f 0.log
    wait
    pkill -9 python3 
    folder_name="kvcache_int8_src_${src_len}_output_${dec_len}_bs_${BATCH_NUM}"
    mkdir $folder_name
    mv *.log $folder_name
    echo "Run End $folder_name paddle token ${paddlet_array[$i]}"
done
