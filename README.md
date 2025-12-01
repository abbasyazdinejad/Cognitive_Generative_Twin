# Cognitive_Generative_Twin

This repository contains the reference implementation of the **Cognitive Generative Twin (CGT)** framework for industrial cyber-physical systems (CPS).  
The code integrates:

- Time-series **preprocessing and windowing** for ICS datasets (e.g., SWaT),
- Deep anomaly detection / forecasting models,
- Evaluation scripts for detection performance and robustness,
- Config-driven experiments for reproducible results.

The project is written in Python and is designed to be **config-driven**: almost all experimental settings (data paths, window size, model type, training parameters) are controlled via YAML/JSON config files under `configs/`.

---

## 1. Project Structure

```text
Cognitive_Generative_Twin/
│
├── configs/          # Experiment/config files (data paths, model + training hyper-params)
├── data/
│   ├── preprocessing.py   # Data loaders & preprocessing (e.g., SWaTPreprocessor)
│   ├── device.py          # Device / sensor utilities (if used)
│   └── ...                # (Optional) helpers for raw → processed data
│
├── evaluation/       # Evaluation scripts (metrics, plots, comparison utilities)
├── models/           # Model definitions (encoders, decoders, generative modules, etc.)
├── scripts/          # Helper scripts (running batches of experiments, plotting, etc.)
├── training/         # Training loops, losses, schedulers, logging utilities
├── utils/            # Common helpers (config parsing, logging, random seeds, etc.)
│
├── main.py           # Main entry point (often for end-to-end run / demo)
├── train.py          # Training launcher using configs
├── requirements.txt  # Python dependencies
├── setup.py          # Optional package installation script
└── README.md
