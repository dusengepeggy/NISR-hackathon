# NISR 2026 Hackathon: household poverty proxy model (Track 2)

Two-head residual MLP (poverty flag + log consumption) on the NISR EICV7 2023-24 survey.
The repo contains code only. Download the raw files yourself from the NISR microdata
catalog (free account) into `data/raw/`; do not commit them.

## Run order
1. `pip install -r requirements.txt`
2. `python prepare_data.py --inspect` and edit `config.py` so column names match (weights, consumption).
3. `python prepare_data.py` builds `data/processed/model_table.csv` (prints every column the leakage guard dropped; read that list).
4. Placeholder run: `python train.py --epochs 1 --run-name placeholder_1epoch`
5. `tensorboard --logdir runs`  (Colab: `%load_ext tensorboard` then `%tensorboard --logdir runs`)
6. Baselines: `python baselines.py`

No data yet? `python prepare_data.py --synthetic` makes fake data so you can check the pipeline runs.
Synthetic results mean nothing; never report them.
