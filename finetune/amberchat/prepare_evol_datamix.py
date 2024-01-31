import json
from random import shuffle
from datasets import load_dataset


output_file: str = "evol-mix.json"
wizard_data = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
sharegpt_data = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")
data_mix = list(sharegpt_data) + [
    {"id": data["idx"], "conversations": data["conversations"]}
    for data in wizard_data
]
shuffle(data_mix)
with open(output_file, 'w') as f:
    json.dump(data_mix, f)
