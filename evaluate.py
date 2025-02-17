# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cruxeval.constants import DATA_DIR, EVAL_DIR
from cruxeval.utils_general import evaluate_score, pass_at_k


def evaluate_generations(generations: dict[str, list], mode):
    # Load the samples
    with open(DATA_DIR / "cruxeval.jsonl", "r") as fp_:
        dataset = [json.loads(l) for l in fp_.readlines()]
    references = [(doc["code"], doc["input"], doc["output"]) for doc in dataset]

    # Run the samples
    try:
        generations_list = [generations[f"sample_{i}"] for i in range(len(dataset))]
    except Exception:  # pylint: disable=broad-exception-caught
        assert (
            False
        ), "check format of generations, should be dictionary of lists with keys of id's in the form sample_i"

    with ProcessPoolExecutor() as executor:
        args_list = zip(generations_list, references, [mode] * len(generations_list))
        results = executor.map(evaluate_score, args_list)
    all_scores = list(results)

    # Compute pass@k scores
    pass_at_1s, pass_at_5s = [], []
    for execution_result in all_scores:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
        pass_at_5s.append(pass_at_k(n, c, 5))

    return {
        "raw_generations": generations,
        "raw_scored_generations": {f"sample_{i}": all_scores[i] for i in range(len(dataset))},
        "pass_at_1": sum(pass_at_1s) / len(pass_at_1s) * 100,
        "pass_at_5": sum(pass_at_5s) / len(pass_at_5s) * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations-path", help="JSON path containing outputs to evaluate.", type=str)
    parser.add_argument("--scored-path", help="path to dump scored results(*.json file)", type=str, default=None)
    args = parser.parse_args()

    if args.scored_path is None:
        parent = Path(args.generations_path).parent.name
        args.scored_path = EVAL_DIR / parent / "scores.json"
    Path(args.scored_path).parent.mkdir(exist_ok=True, parents=True)

    if Path(args.scored_path).exists():
        print(f"Already exists path: {args.scored_path}. Scoring skip")
        sys.exit(0)

    with open(args.generations_path, "r") as fp:
        _generations = json.load(fp)
    print(f"Scoring {args.generations_path}... expect around a minute")

    results_ = evaluate_generations(
        generations=_generations, mode="input" if "input" in args.generations_path else "output"
    )
    print("Finished!")
    print("pass@1:", round(results_["pass_at_1"], 1), "pass@5:", round(results_["pass_at_5"], 1))
    print(f"Dumping to {args.scored_path}")
    with open(args.scored_path, "w") as fp:
        json.dump(results_, fp)
