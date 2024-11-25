#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=true

output_lens=(1024)
models=("Meta-Llama-3.1-8B-Instruct")
use_flashinfer=(1)


if [ $# -eq 0 ]; then
  save_path="results/throughput"
  echo "No save path provided, using default path: $save_path"
else
  save_path=$1
fi

mkdir -p "$save_path"


for output_len in "${output_lens[@]}"; do
    for model in "${models[@]}"; do
        for flashinfer_flag in "${use_flashinfer[@]}"; do

            if [ $flashinfer_flag -eq 1 ]; then
                export VLLM_ATTENTION_BACKEND="FLASHINFER"
                tag="w_flashinfer"
            else
                tag="wo_flashinfer"
            fi

            model_path="/models/${model}"
            save_name="${model}_outlen${output_len}_${tag}"
            echo -e "\n\n\n######## Running Exp ${save_name} ########\n\n\n"
            
            command="python ./benchmark_throughput.py \
                --backend vllm \
                --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
                --model ${model_path} \
                --tokenizer ${model_path} \
                --output-json ${save_path}/${save_name}.json \
                --trust-remote-code \
                -tp 4 --output-len ${output_len} \
                --max-model-len 4096"

            eval $command

            unset VLLM_ATTENTION_BACKEND
            sleep 5
        done
    done
done

