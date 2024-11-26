#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="FLASHINFER"

input_lens=(4096)
batch_size=(1)
models=("Meta-Llama-3.1-8B-Instruct")
use_fp16=(1 0)


if [ $# -eq 0 ]; then
  save_path="results/latency"
  echo "No save path provided, using default path: $save_path"
else
  save_path=$1
fi

mkdir -p "$save_path"


for input_len in "${input_lens[@]}"; do
    for bs in "${batch_size[@]}"; do
        for model in "${models[@]}"; do
            for fp16_flag in "${use_fp16[@]}"; do

                if [ $fp16_flag -eq 1 ]; then
                    export VLLM_USE_FLASHINFER_FP16_ACCUM=1
                    tag="w_flashinfer_fp16"
                else
                    tag="wo_flashinfer_fp16"
                fi

                model_path="/models/${model}"
                save_name="${model}_inlen${input_len}_outlen1_bs${bs}_${tag}"
                echo -e "\n\n\n######## Running Exp ${save_name} ########\n\n\n"
                
                # command="ncu -k regex:\\(\\?i\\).*prefill.* --set full --replay-mode kernel \
                #     -f -o ${save_name} --devices 0 \
                #     python ./benchmark_latency.py \
                #     --input-len ${input_len} --output-len 1 --batch-size ${bs} \
                #     --num-iters-warmup 0 --num-iters 1 \
                #     --model ${model_path} -tp 4 --max-model-len 4096 \
                #     --trust-remote-code --dtype float16 \
                #     --output-json ${save_path}/${save_name}.json"  

                command="python ./benchmark_latency.py \
                    --input-len ${input_len} --output-len 1 --batch-size ${bs} \
                    --num-iters-warmup 0 --num-iters 1 \
                    --model ${model_path} -tp 4 --max-model-len 8192 \
                    --trust-remote-code --dtype float16 \
                    --output-json ${save_path}/${save_name}.json"    
                eval $command

                unset VLLM_USE_FLASHINFER_FP16_ACCUM
                sleep 5
            done
        done
    done
done

