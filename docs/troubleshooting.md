# Troubleshooting

## GLFW or display errors

If you see `RuntimeError: Failed to create window`, the simulation tried to open a viewer without a working display. Use a simulation config with:

```yaml
headless: true
```

The demo config uses `config/_SIMULATION/simulation_headless_demo.yaml`.

## Missing FARMS packages

Install the local FARMS packages before running demos:

```bash
bash setup.sh
```

## Log directory already exists

By default, runs write to `experiments/<experiment_id>/logs/<date>/<seed>`. Existing non-debug log directories abort to avoid overwriting results. For quick local demos, use seed `999` or a fresh `-d` value.

## Short training does not create `best_model.zip`

The public-demo training path now saves `last_model_trained.zip` and falls back to `best_model.zip` after short runs where the evaluator did not produce a best checkpoint.
