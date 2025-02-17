# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
from pathlib import Path

from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()

    directory = Path(args.eval_dir)

    accs = {}
    models = []
    for file in directory.rglob("*.json"):
        with open(file, "r") as fp:
            f = json.load(fp)
        parent = file.parent.name
        model_name = parent.split("_temp")[0].strip()
        temperature = float(parent.split("_temp")[1].split("_")[0])
        mode = parent.split("_")[-1]
        models.append(model_name)

        if temperature == 0.2:
            accs[(mode, model_name, temperature)] = round(f["pass_at_1"], 1)
        else:
            accs[(mode, model_name, temperature)] = round(f["pass_at_5"], 1)

    models = list(set(models))
    models.sort()

    for mode in ["input", "output"]:
        data = []
        for model in models:
            try:
                pass_at_1 = accs[(mode, model, 0.2)]
            except Exception:  # pylint: disable=broad-exception-caught
                pass_at_1 = "n/a"
            try:
                pass_at_5 = accs[(mode, model, 0.8)]
            except Exception:  # pylint: disable=broad-exception-caught
                pass_at_5 = "n/a"
            try:
                data.append([model, pass_at_1, pass_at_5])
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        headers = ["Model", "Pass@1", "Pass@5"]
        data.sort(key=lambda x: x[1])
        table = tabulate(data, headers=headers, tablefmt="pipe")
        print(f"********* CRUXEval-{mode.capitalize()} *********\n")
        print(table)
        print("\n")


if __name__ == "__main__":
    main()
