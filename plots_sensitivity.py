from stable_baselines3 import PPO
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import conf

# gradient for plotting
gradient_2 = [
    "#072140",
    "#0A2D57",
    "#0E396E",
    "#114584",
    "#14519A",
    "#165DB1",
    "#5E94D4",
    "#9ABCE4",
    "#C2D7EF",
    "#FED702",
    "#CBAB01",
    "#FEDE34",
    "#FEE667",
    # "#F7811E",
    # "#D99208",
    # "#F9BF4E",
    # "#FAD080",
    # "#B55CA5",
    # "#9B468D",
    # "#C680BB",
    # "#D6A4CE",
    # "#E6C7E1",
    # "#8F81EA",
    # "#6955E2",
    # "#B6ACF1",
    # "#C9C2F5",
    "#9FBA36",
    "#7D922A",
    "#B6CE55",
    "#C7D97D",
    # "#333A41",
    # "#475058",
    # "#6A757E",
]


# model and CORRECT experiment_id to load
model_path = "experiments/415/logs/16-06-2023_09:00:03/best_model.zip"
# model_path = "experiments/999/logs/20-06-2023_08:20:34/best_model.zip"
experiment_id = "415"
# experiment_id = "999"

# base observation upon which the input neurons are varied
base_obs = torch.tensor(
    [
        [
            -1.3014,
            0.2440,
            -0.5538,
            -0.1316,
            -1.1946,
            -1.6352,
            -2.4294,
            -0.9034,
            -0.3155,
            1.2889,
            0.9747,
            1.6770,
            -1.0865,
            -1.3532,
            1.0439,
            -1.0884,
            1.0071,
            -0.3786,
            -0.0625,
            -0.6335,
        ]
    ]
)

# base_obs = torch.tensor(
#     [
#         [
#             1.4785,
#             1.1418,
#             1.6578,
#             1.6337,
#             1.7156,
#             0.8025,
#             0.2714,
#             -0.8171,
#             0.1219,
#             -1.1453,
#             -0.0180,
#             -1.3139,
#             -1.0594,
#             0.1475,
#             0.4937,
#             0.9981,
#             -0.8572,
#             1.5013,
#             -0.6923,
#             1.4486,
#         ]
#     ]
# )

# base_obs = torch.tensor([([0.0] * 20)])

with open(f"./experiments/{experiment_id}/" + "conf.yaml") as experiment_config:
    conf.init(experiment_config, experiment_id)
model = PPO.load(model_path)
plots = {}


# perform gradient analysis for one specific observation
# conf.CONF["misc"]["log_grads"] will contain the gradients of all outputs wrt to the inputs
conf.CONF["misc"]["log_grads"] = True
obs = base_obs
action, _ = model.predict(obs, deterministic=True)
grads = conf.CONF["misc"]["log_grads"]
for i in range(len(grads)):
    fig = plt.figure(f"Gradients of output neuron {i} w.r.t. input neurons")
    plt.bar([j for j in range(len(grads[i][0]))], grads[i][0], color="#072140")
    plt.xticks([j for j in range(len(grads[i][0]))])
    plt.xlabel("Input neuron")
    plt.ylabel(f"Gradient")
    plt.grid(True)
    plots[f"Gradients of output neuron {i} w.r.t. input neurons"] = fig
conf.CONF["misc"]["log_grads"] = False

# save all plots in one pdf
with PdfPages(
    f"plot_utils/network_sensitivity_analysis_plots/grads_{experiment_id}.pdf"
) as pdf:
    for name, plot in plots.items():
        plot.suptitle(name)
        pdf.savefig(plot.figure)  # pdf
print("Saved grad plots to pdf.")

plots = {}

# obs of shape (1, 20)
# features: 0-9: joint positions; 10-19: phases
# normalized input values through normalization wrapper should be roughly in range [-2, 2]
inputs = np.arange(-5.0, 5.0, 0.01)
input_neurons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# plot all output neurons for varying one input neuron
for input_neuron in input_neurons:
    actions = []
    # iterate over input values
    obs = base_obs.clone()
    pass

    for i in inputs:
        obs[0][input_neuron] = i
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

    fig = plt.figure(f"Varying input neuron {input_neuron}")
    for i in range(9):
        plt.plot(
            inputs,
            [action[0][i] for action[0] in actions],
            label=f"Output neuron {i}",
            # color=gradient_2[i * 2],
        )
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )
    plt.xlabel("Input value")
    plt.ylabel("Output value")
    # plt.ylim(-1.1, 1.1)
    plt.grid(True)

    plots[f"Varying input neuron {input_neuron}"] = fig

# plot one output neuron for varying all input neurons
for output_neuron in range(9):
    sensitivity_sums = []
    fig = plt.figure(f"output neuron {output_neuron}")
    for j, input_neuron in enumerate(input_neurons):
        # iterate over input values
        obs = base_obs.clone()
        actions = []

        for i in inputs:
            obs[0][input_neuron] = i
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action[0][output_neuron])

        sensitivity_sums.append(sum(np.abs(actions)))

        plt.plot(
            inputs,
            actions,
            label=f"input neuron {input_neuron}",
            #   color=gradient_2[j],
        )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    # plt.ylim(-1.1, 1.1)
    ax = plt.gca()
    plt.grid(True)

    plots[f"output neuron {output_neuron}"] = fig

    # new figure for sensitivity sums
    fig = plt.figure(f"Sum of sensitivities for output neuron {output_neuron}")
    plt.bar([j for j in range(len(input_neurons))], sensitivity_sums, color="#072140")
    plt.xticks([j for j in range(len(input_neurons))])
    plt.xlabel("Input neuron")
    plt.ylabel(f"Sum of sensitivities")
    plt.grid(True)
    plots[f"Sum of sensitivities for output neuron {output_neuron}"] = fig

# save all plots in one pdf
with PdfPages(
    f"plot_utils/network_sensitivity_analysis_plots/sensitivity_{experiment_id}.pdf"
) as pdf:
    for name, plot in plots.items():
        plot.suptitle(name)
        pdf.savefig(plot.figure, bbox_inches="tight")  # pdf
print("checkpoint 2")
print("Saved sensitivity plots to pdf.")
