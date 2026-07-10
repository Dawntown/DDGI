# DDGI

DDGI is a Python package for identifying driver genes from single-cell perturbation datasets. It trains a neural model on paired control and perturbed cells, then ranks candidate perturbation genes for each target cell or condition.

## Installation

DDGI is intended to run with CUDA. First check the CUDA version supported by your NVIDIA driver:

```bash
nvidia-smi
```

Create a conda environment:

```bash
conda create -n ddgi python=3.10 -y
conda activate ddgi
```

Install a CUDA-enabled PyTorch build that is compatible with your driver. For example, if `nvidia-smi` reports CUDA 12.9, install the CUDA 12.8 PyTorch wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If your driver only supports an older CUDA runtime, choose a matching PyTorch wheel instead, for example:

```bash
# CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

The last line should print `True`. Then install DDGI and the remaining dependencies:

```bash
git clone https://github.com/Dawntown/DDGI.git
cd DDGI/
pip install -r requirements.txt
pip install -e .
```

If you see an error such as `The NVIDIA driver on your system is too old`, reinstall PyTorch with a CUDA wheel no newer than the CUDA version shown by `nvidia-smi`.

## Quick Start

Prepare the tutorial dataset:

```bash
python tutorial/prepare_tutorial_data.py
```

This script will help to download example anndata files from [CellNavi's repository](https://github.com/DLS5-Omics/CellNavi) and split CSVs for following running DDGI tutorial. The tutorial dataset will be saved in `datasets/schmidt_tutorial/`.

Open the tutorial notebook in vscode or jupyter lab:

```bash
tutorial/run.ipynb
```

The tutorial uses `tutorial/config_tutorial.yaml`, reads data from `datasets/schmidt_tutorial/`, and writes model outputs to `results/`.

## Python Usage

```python
from driver_genes.pipelines import Pipeline

pipe = Pipeline(config_path="tutorial/config_tutorial.yaml")
pipe.fit()
pipe.save_best_model()
metrics, predictions = pipe.evaluate(num_results=1, seed=0)
```

## Input Data

DDGI expects:

1. An `.h5ad` file with expression values in `adata.X` or a selected layer, and perturbation labels in `adata.obs`.
2. A split CSV with `cell` and `split` columns. Optional `subsplit` values are used to report metrics for named groups.

By default, perturbation labels are read from `adata.obs["perturbation"]`, and control cells are labeled `control`.
