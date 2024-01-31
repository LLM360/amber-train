import datasets
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Union, Tuple


def get_saferlhf(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading PKU-SafeRLHF dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('PKU-Alignment/PKU-SafeRLHF', split=split, cache_dir=cache_dir)
    filtered_dataset = dataset.filter(lambda ex: ex["is_response_0_safe"] != ex["is_response_1_safe"])
    print('done')

    def split_prompt_and_responses(ex):
        prompt = ex['prompt']
        chosen, rejected = ex["response_0"], ex["response_1"]
        if ex["is_response_1_safe"]:
            chosen, rejected = rejected, chosen
        return prompt, chosen, rejected

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(filtered_dataset, desc='Processing PKU-SafeRLHF', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data
