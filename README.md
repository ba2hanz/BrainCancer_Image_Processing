# Brain Cancer Classification

Deep learning project for classifying brain tumor MRI images into `Brain_Glioma`, `Brain_Menin`, and `Brain_Tumor`.

## Project structure
- `data/brain_cancer_data/` – train/validation/test folders (NOT in repo)
- `data_preprocess.py` – Keras Sequence for multi-input models
- `model_def.py` – Transfer, hybrid, and custom model definitions
- `main.py` – Training, evaluation, and plotting

## Setup (recommended: Conda)
```bash
# Create and activate env
conda create -n brain-cancer python=3.11 -y
conda activate brain-cancer

# Install dependencies
pip install -r requirements.txt
```

GPU (optional): Use Conda to manage CUDA/cuDNN easily:
```bash
conda install -c conda-forge tensorflow-gpu -y
```

## Data
Place data in:
```
./data/brain_cancer_data/
  train/
    Brain_Glioma/
    Brain_Menin/
    Brain_Tumor/
  validation/
    ...
  test/
    ...
```

## Run training
```bash
python main.py
```
Outputs:
- `results/plots/*`
- `best_weights/*.keras`
- `best_model/*.keras`

## Notes
- Large data and outputs are git-ignored by default.
- Custom model input size is 512x512.
- For Windows GPU, prefer Conda-based TensorFlow GPU.

