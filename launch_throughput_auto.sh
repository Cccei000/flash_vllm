#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="FLASHINFER"

output_lens=(128 1024)
models=("Meta-Llama-3.1-8B-Instruct")
use_fp16=(1 0)


if [ $# -eq 0 ]; then
  save_path="results/throughput"
  echo "No save path provided, using default path: $save_path"
else
  save_path=$1
fi

mkdir -p "$save_path"


for output_len in "${output_lens[@]}"; do
    for model in "${models[@]}"; do
        for fp16_flag in "${use_fp16[@]}"; do

            if [ $fp16_flag -eq 1 ]; then
                export VLLM_USE_FLASHINFER_FP16_ACCUM=1
                tag="w_flashinfer_fp16"
            else
                tag="wo_flashinfer_fp16"
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
                --max-model-len 4096"

            eval $command

            unset VLLM_USE_FLASHINFER_FP16_ACCUM
            sleep 5
        done
    done
done

