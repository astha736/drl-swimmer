#!/usr/bin/env python3
import argparse
import os

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--eval_path", required=True)
    return parser.parse_args()


def parse_metric_list(value):
    parsed = value.replace("[", "").replace("]", "").replace(" ", "").split(",")
    return [float(parsed[0]), float(parsed[1])]


def main():
    args = parse_args()

    results_path = f"{args.eval_path}/results"
    results = {}
    seeds = []

    os.makedirs(results_path, exist_ok=True)

    subdirs = [f.name for f in os.scandir(args.eval_path) if f.is_dir()]

    metrics_to_skip = ["0_n_eval_episodes", "0_data_format"]
    subdirs_to_skip = ["results", "999"]

    for subdir in subdirs:
        if subdir in subdirs_to_skip:
            continue

        try:
            with open(
                f"{args.eval_path}/{subdir}/eval_metrics.yaml", encoding="utf-8"
            ) as f:
                eval_metrics = yaml.full_load(f)
        except FileNotFoundError:
            print(f"    Eval file for subdir not found: {subdir}")
            continue

        try:
            for metric in eval_metrics:
                if metric in metrics_to_skip:
                    continue
                if metric == "0_seed":
                    seeds.append(eval_metrics[metric])
                    continue

                eval_metrics[metric] = parse_metric_list(eval_metrics[metric])
                if metric not in results:
                    results[metric] = []
                results[metric].append(eval_metrics[metric][0])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            print(f"    Could not parse eval metrics for subdir {subdir}: {exc}")

    for result in results:
        results[
            result
        ] = f"[{np.mean(results[result]):.4f}, {np.std(results[result]):.4f}, {np.min(results[result]):.4f}, {np.max(results[result]):.4f}]"
    results["0_data_format"] = "[mean, std, min, max]"
    results["0_seeds"] = f"{seeds}"

    with open(f"{results_path}/results.yaml", "w", encoding="utf-8") as f:
        yaml.dump(results, f)


if __name__ == "__main__":
    main()
