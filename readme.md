# Diffcast and AlphaPre Forecasting Models

This repository contains the code for training and evaluating precipitation nowcasting models, including **Diffcast** and **AlphaPre**. It also supports other deterministic models like **ConvLSTM** and **SimVP**.

---

## ⚙️ Environment Setup

To run the models, first set up the required Conda environment.


1.  **Create the Conda environment from the `env.yaml` file:**
    ```bash
    conda env create -f env.yaml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate diffcast
    ```

---

## 💾 Datasets

The models are trained and evaluated on several weather radar datasets. The data is expected to be located at the following server addresses.

* **SEVIR Dataset:**
    `vatsal@10.24.52.210:/qdata/vatsal/Data/Datasets/sevir/`

* **VIL (Vertically Integrated Liquid) Dataset:**
    `vatsal@10.24.52.210:/qdata/vatsal/Data/Datasets/VIL/VIL_scaled_0_255/full_data`



## 🚀 Running the Models

Instructions for training and evaluating the models are provided below.

### Diffcast

* **To Train the model:**
    ```bash
    python3 run_diffcast.py --use_diff
    ```

* **To Evaluate/Test the model:**
    You must provide the path to a saved model checkpoint using the `--ckpt_milestone` argument.
    ```bash
    python3 run_diffcast.py --eval --use_diff --ckpt_milestone <address_of_checkpoint>
    ```

### AlphaPre & Other Deterministic Models

The `run_alphapre_convlstm.py` script is used to run **AlphaPre** as well as other deterministic models like **ConvLSTM**, **SimVP**, etc.

* **To Train the model:**
    ```bash
    python3 run_alphapre_convlstm.py
    ```

* **To Evaluate/Test the model:**
    Provide the path to a saved model checkpoint using the `--ckpt_milestone` argument.
    ```bash
    python3 run_alphapre_convlstm.py --eval --ckpt_milestone <address_of_checkpoint>
    ```

---
### Important Note for VIL Dataset

When running experiments with the VIL dataset, you must specify the path to the file containing the rainy day sequences. Use the `--file_rain_seq_add` argument to point to the correct file located in the `Rainy_days_file` folder.

---

## 🔧 Command-Line Arguments

Each run script (`run_diffcast.py`, `run_alphapre_convlstm.py`) contains various command-line arguments that can be used to customize the model's architecture, training parameters, and data handling. Please inspect the files for a full list of available options.

## Pretrained models download links

Diffcast: https://drive.google.com/file/d/1y8BvYz3U_awm1eAYqXBy6tgbMy8t40Xr/view?usp=sharing

Alphapre: https://drive.google.com/file/d/1hzT2-biQhWuKTER8w1yoQx5Zh0nMYl80/view?usp=sharing
