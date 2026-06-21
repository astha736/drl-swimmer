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

Run a short CPG/feedback architecture test:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

Run a tiny PPO training demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_short_train -d demo -s 999
```

The demo training is intentionally small. It checks that the pipeline runs; it is not expected to reproduce paper-quality swimming.
