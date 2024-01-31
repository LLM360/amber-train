TRAIN_SCRIPT_PATH=direct-preference-optimization/train.py

python -u $TRAIN_SCRIPT_PATH \
    model=blank_model \
    model.name_or_path=LLM360/AmberChat \
    model.block_name=LlamaDecoderLayer \
    datasets=[saferlhf] \
    loss=sft \
    exp_name=saferlhf_sft_ambersafe \
    max_length=512 \
    max_prompt_length=256 \
    gradient_accumulation_steps=2 \
    batch_size=64 \
    eval_batch_size=32 \
    n_epochs=3 \
    lr=2e-7 \
    trainer=FSDPTrainer \
    sample_during_eval=false
