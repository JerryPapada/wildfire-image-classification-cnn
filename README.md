# Wildfire Image Classification with CNNs

TensorFlow/Keras experiments for classifying wildfire imagery. The repository includes CNN/model helpers, training and evaluation code, image-resizing and split utilities, hyperparameter work, and a project report.

## Workflow components

- model.py and model_functions.py define model construction and training/evaluation helpers.
- main.py orchestrates the transfer-learning experiments.
- resize.py prepares image dimensions for the model workflow.
- split_folder.py creates dataset splits from image folders.
- hyperparameters.py and hyperparameter_tuning/ hold tuning configuration and experiments.

## Requirements and data

Install packages from requirements.txt in a Python environment compatible with its TensorFlow/Keras version. The image dataset is not included at the repository root. main.py contains machine-specific absolute paths to train/validation/test folders; edit the data-path configuration to local directories before running. No verified dataset download command is provided.

## Run

After preparing those folders and checking the preprocessing assumptions:

```bash
python3 main.py
```

The resize and split utilities are separate preprocessing steps; inspect their expected input/output directory structure before using them. Keep dataset partitions separate when evaluating a model.

## Credential warning

main.py and model.py contain hard-coded Telegram bot credentials. Treat them as exposed and rotate/revoke them. The training workflow should not require a Telegram secret to be committed; do not publish replacement values. No model accuracy or dataset-independent result is claimed by this README.