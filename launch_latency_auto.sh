#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=true

output_lens=(1)
input_lens=(1024)
batch_size=(2)
model="Llama-2-7b-chat-hf"
use_flashinfer=(1)


if [ $# -eq 0 ]; then
  save_path="results/latency"
  echo "No save path provided, using default path: $save_path"
else
  save_path=$1
fi

mkdir -p "$save_path"


for input_len in "${input_lens[@]}"; do
    for output_len in "${output_lens[@]}"; do
        for bs in "${batch_size[@]}"; do
            for flashinfer_flag in "${use_flashinfer[@]}"; do

                if [ $flashinfer_flag -eq 1 ]; then
                    export VLLM_ATTENTION_BACKEND="FLASHINFER"
                    tag="w_flashinfer"
                else
                    tag="wo_flashinfer"
                fi

                model_path="/models/${model}"
                save_name="${model}_inlen${input_len}_outlen${output_len}_bs${bs}_${tag}"
                echo -e "\n\n\n######## Running Exp ${save_name} ########\n\n\n"
                
                command="ncu -k regex:\\(\\?i\\).*prefill.* --set full --replay-mode kernel \
                    -f -o ${save_name}_reduction --devices 0 \
                    python ./benchmark_latency.py \
                    --input-len ${input_len} --output-len ${output_len} --batch-size ${bs} \
                    --num-iters-warmup 0 --num-iters 1 \
                    --model ${model_path} -tp 4 --max-model-len 4096 \
                    --trust-remote-code --dtype float16 \
                    --output-json ${save_path}/${save_name}.json"
    
                eval $command

                unset VLLM_ATTENTION_BACKEND
                sleep 5
            done
        done
    done
done

