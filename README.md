# Wildfire Image Classification with CNNs

TensorFlow/Keras experiments for classifying wildfire imagery. The repository contains model/training code, preprocessing utilities, hyperparameter scripts, and a report.

## Main files

- `model.py` and `model_functions.py`: model construction, training, and evaluation helpers.
- `main.py`: experiment entry point using transfer-learning backbones.
- `resize.py` and `split_folder.py`: image preparation utilities.
- `hyperparameters.py` and `hyperparameter_tuning/`: tuning configuration and work.

## Requirements and data

Install the packages listed in `requirements.txt` in a compatible Python environment. The expected train/validation/test image directories are not included in the repository's top-level tree. The inspected `main.py` uses machine-specific absolute dataset paths; update them to local dataset directories before running.

## Run

After configuring the data paths and reviewing the notification behavior:

```bash
python3 main.py
```

The training code also contains hard-coded Telegram bot credentials. Treat them as exposed and rotate them; do not reuse or publish replacements. The repository does not provide a verified dataset download command or claim any model accuracy here.
