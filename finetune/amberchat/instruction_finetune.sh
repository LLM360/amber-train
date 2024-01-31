TRAIN_SCRIPT_PATH=FastChat/fastchat/train/train_mem.py
MODEL_NAME=LLM360/Amber
DATA_PATH=evol-mix-split.json
OUTPUT_DIR=amberchat_ckpts/

torchrun --nproc_per_node=8 --master_port=20001 $TRAIN_SCRIPT_PATH \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
