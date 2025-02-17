# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import logging

from cruxeval.constants import DATA_DIR, DUMP_DIR
from cruxeval.prompts import (
    batch_prompt_cot_input,
    batch_prompt_cot_output,
    batch_prompt_direct_input,
    batch_prompt_direct_output,
)

logging.basicConfig(filename="log.log", filemode="w", level=logging.INFO)


def run_giga(model, mode, cot, temperature, parallel):
    with open(DATA_DIR / "cruxeval.jsonl", "r") as fp:
        dataset = [json.loads(l) for l in fp.readlines()]

    save_dir = get_save_dir(mode, model, cot, temperature)

    if mode == "input":
        prompts = [(data["code"], data["output"]) for data in dataset]
    else:
        prompts = [(data["code"], data["input"]) for data in dataset]

    if cot:
        max_tokens = 1000
    else:
        max_tokens = 100

    fn = {
        (True, "input"): batch_prompt_cot_input,
        (True, "output"): batch_prompt_cot_output,
        (False, "input"): batch_prompt_direct_input,
        (False, "output"): batch_prompt_direct_output,
    }[(cot, mode)]

    outputs = fn(prompts, temperature=temperature, n=10, model=model, max_tokens=max_tokens, parallel=parallel)
    outputs_dict = {f"sample_{i}": [j[0] for j in o] for i, o in enumerate(outputs)}
    with open(save_dir, "w") as fp:
        json.dump(outputs_dict, fp)
    return outputs


def get_save_dir(mode, model, cot, temperature):
    directory = DUMP_DIR / f"{model}{'+cot' if cot else ''}_temp{temperature}_{mode}"
    directory.mkdir(exist_ok=True, parents=True)
    return directory / "generations.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2, choices=[0.2, 0.8])
    parser.add_argument("--mode", type=str, default="input", choices=["input", "output"])
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    run_giga(args.model, args.mode, args.cot, args.temperature, args.parallel)
