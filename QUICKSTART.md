# Quickstart

Install dependencies and local FARMS packages:

```bash
python3 -m venv obstacle
source obstacle/bin/activate
pip install -r requirements.txt
bash setup.sh
```

Run a headless smoke test:

```bash
python scripts/smoke_test.py
```

Run the bio-inspired caudal stretch-feedback demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_caudal_feedback_viewer -d demo -s 999
```

Run the open-loop CPG swimming demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_cpg_swim_viewer -d demo -s 999
```

These viewer demos need a working display/OpenGL setup. On a headless server,
use `demo_arch_test` instead.

Run a tiny PPO training demo that learns stretch-feedback weights from an
initially open/decoupled network:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_drl_feedback_train -d demo -s 999
```

The demo training is intentionally small. It checks that the pipeline runs; it is not expected to reproduce paper-quality swimming.
