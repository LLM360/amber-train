TRAIN_SCRIPT_PATH=direct-preference-optimization/train.py
SFT_CKPT_PATH=.cache/user/saferlhf_sft_ambersafe/LATEST/policy.pt

python -u $TRAIN_SCRIPT_PATH \
    model=blank_model \
    model.name_or_path=LLM360/AmberChat \
    model.archive=$SFT_CKPT_PATH \
    model.block_name=LlamaDecoderLayer \
    datasets=[saferlhf] \
    loss=dpo \
    loss.beta=0.1 \
    exp_name=saferlhf_dpo_ambersafe \
    max_length=512 \
    max_prompt_length=256 \
    gradient_accumulation_steps=4 \
    batch_size=32 \
    eval_batch_size=32 \
    n_epochs=3 \
    lr=5e-7 \
    trainer=FSDPTrainer \
    sample_during_eval=false
