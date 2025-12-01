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
## 2. Installation

Clone the repository

git clone https://github.com/abbasyazdinejad/Cognitive_Generative_Twin.git
cd Cognitive_Generative_Twin


Create and activate a virtual environment (recommended)

python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate


Install dependencies

pip install --upgrade pip
pip install -r requirements.txt


If you prefer, you can also install the project as a package:

pip install -e .

## 3. Datasets & Preprocessing

The repository is set up to work with industrial control system time-series data such as SWaT (and can be extended to other datasets).

Place raw data

Put the raw SWaT (and other) files under the directory expected by your config, e.g.:

data/raw/swat/
    Normal.xlsx
    Attack.xlsx
    ...


The exact filenames and paths are configured in configs/*.yaml and used by data/preprocessing.py.

Preprocessing

The class SWaTPreprocessor in data/preprocessing.py:

loads the raw normal / attack Excel files,

drops timestamp columns (if present),

maps labels to {0 = normal, 1 = attack},

keeps only numeric columns,

applies sliding windows (window_size, stride) from the config.

You can either:

run preprocessing on the fly when calling train.py (most configs do this), or

write a small script in scripts/ that imports SWaTPreprocessor and saves processed numpy/CSV files.

## 4. Running Experiments

Most experiments are launched via train.py using a configuration file.

4.1 Basic training run
python train.py --config configs/swat_baseline.yaml


Typical config fields:

data: paths, window size, stride, train/val/test split,

model: model type and hyper-parameters,

training: epochs, batch size, optimizer, learning rate, scheduler,

logging: output directory, checkpoints, etc.

(Adjust the exact config filename to match the files you have under configs/.)

4.2 Evaluation

After training, use the scripts in evaluation/ to compute metrics and generate plots, e.g.:

python evaluation/evaluate.py --config configs/swat_baseline.yaml --checkpoint path/to/checkpoint.pt


(Again, replace with the actual script / checkpoint names in your repo.)

## 5. Reproducibility

To ensure experiments are reproducible, the project typically:

fixes random seeds (see helpers in utils/),

keeps all hyper-parameters in a single config file per experiment,

logs metrics and configuration to disk (and optionally TensorBoard / WandB, depending on your setup).

If you add new experiments, create a new config under configs/ rather than changing code.

## 6. Extending the Codebase

You can extend the framework in several ways:

New datasets:
Implement a new preprocessor class in data/preprocessing.py or a new file under data/, then wire it up via a config entry.

New models:
Add your model to models/ and expose it in the model factory used by the training loop.

New evaluation metrics:
Add them under evaluation/ and call them in the evaluation script or training hooks.
