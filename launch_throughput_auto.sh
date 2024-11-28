#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="FLASHINFER"

output_lens=(16)
model="Meta-Llama-3.1-8B-Instruct"
use_fp16=(0 1)
use_tensor_core=(0 1)


if [ $# -eq 0 ]; then
  save_path="results/throughput"
  echo "No save path provided, using default path: $save_path"
else
  save_path=$1
fi

mkdir -p "$save_path"


for output_len in "${output_lens[@]}"; do
    for fp16_flag in "${use_fp16[@]}"; do
        for tensor_core_flag in "${use_tensor_core[@]}"; do

            if [ $fp16_flag -eq 1 ]; then
                export VLLM_USE_FLASHINFER_FP16_ACCUM=1
                if [ $tensor_core_flag -eq 1 ]; then
                    export VLLM_USE_FLASHINFER_DECODE_WITH_PREFILL=1
                    tag="w_fp16_accum_w_tensor_core"
                else
                    tag="w_fp16_accum_wo_tensor_core"
                fi
            else
                if [ $tensor_core_flag -eq 1 ]; then
                    export VLLM_USE_FLASHINFER_DECODE_WITH_PREFILL=1
                    tag="wo_fp16_accum_w_tensor_core"
                else
                    tag="wo_fp16_accum_wo_tensor_core"
                fi
            fi

            model_path="/models/${model}"
            save_name="${model}_outlen${output_len}_${tag}"
            echo -e "\n\n\n######## Running Exp ${save_name} ########\n\n\n"
            
            command="python ./benchmark_throughput.py \
                --backend vllm \
                --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
                --model ${model_path} --dtype float16 \
                --tokenizer ${model_path} \
                --output-json ${save_path}/${save_name}.json \
                --trust-remote-code \
                -tp 4 --output-len ${output_len} \
                --max-model-len 4096 --num-prompts 1"

            $command

            unset VLLM_USE_FLASHINFER_FP16_ACCUM
            unset VLLM_USE_FLASHINFER_DECODE_WITH_PREFILL
            sleep 5
        done
    done
done

