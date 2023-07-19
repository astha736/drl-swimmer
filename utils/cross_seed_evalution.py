#!/usr/bin/env python3
import os
from re import X
import argparse
import yaml
import numpy as np

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--eval_path", required=True)
args = parser.parse_args()

results_path = f"{args.eval_path}/results"
results = {}
seeds = []

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# get all subdirs
subdirs = [f.name for f in os.scandir(args.eval_path) if f.is_dir()]

metrics_to_skip = ["0_n_eval_episodes", "0_data_format"]


for subdir in subdirs:
    if subdir == "results":
        continue
    # read eval results
    with open(f"{args.eval_path}/{subdir}/eval_metrics.yaml") as f:
        eval_metrics = yaml.full_load(f)
        # convert string-list to list
        for metric in eval_metrics:
            if metric in metrics_to_skip:
                continue
            if metric == "0_seed":
                seeds.append(eval_metrics[metric])
                continue
            _list = (
                eval_metrics[metric]
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .split(",")
            )
            eval_metrics[metric] = [float(_list[0]), float(_list[1])]  # [mean, std]
            # collect all mean values
            if not metric in results:
                results[metric] = []
            results[metric].append(eval_metrics[metric][0])

# evaluate statistics on mean values of different seeds
for result in results:
    results[
        result
    ] = f"[{np.mean(results[result]):.4f}, {np.std(results[result]):.4f}, {np.min(results[result]):.4f}, {np.max(results[result]):.4f}]"
results["0_data_format"] = "[mean, std, min, max]"
results["0_seeds"] = f"{seeds}"

with open(f"{results_path}/results.yaml", "w") as f:
    yaml.dump(results, f)
