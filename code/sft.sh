#!/usr/bin/env bash

export WANDB_API_KEY="" # your wandb API key
export WANDB_MODE="online"
export WANDB_PROJECT= # your wandb project name

dataset=cr_sft_train
MODEL_PATH=path/to/Qwen2.5-Coder-7B-Instruct
model_dir=$(dirname "$MODEL_PATH")
model_name=$(basename "$MODEL_PATH")
current_time=$(date +"%Y%m%d_%H%M%S")
stage=sft 
finetuning_type=lora
run_name="${dataset}_${model_name}_${stage}_${finetuning_type}_${current_time}"

llamafactory-cli train \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage ${stage} \
    --do_train \
    --finetuning_type ${finetuning_type} \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target all \
    --quantization_bit 4 \
    --quantization_method bnb \
    --dataset ${dataset} \
    --template qwen \
    --cutoff_len 8192 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 32 \
    --output_dir saves/${WANDB_PROJECT}/${model_name}/${finetuning_type}/${stage}/${current_time} \
    --logging_steps 5 \
    --save_steps 20 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --run_name ${run_name} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
     --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 2>&1 | tee "logs/${run_name}.log"