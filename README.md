# Handwritten Digit Recognition (PyTorch)

Simple, fast MNIST digit classifier using a CNN with batch normalization and dropout. Trains to ≥99% validation accuracy and ~99% test accuracy.

## Quick Start

- Install: `pip install -r requirements.txt`
- Train: `python train.py`
- Outputs: `digit_recognition_cnn.pth`, `training_history.png`, `predictions.png`

## Model

- 3 conv blocks: `Conv → BN → ReLU → Pool → Dropout` (channels 32→64→128)
- Dense layers with BN + Dropout; logits for 10 classes
- Light augmentation: rotation/translation; normalized with MNIST mean/std

## Results

- Validation: 99.00%
- Test: 99.08%
- Early stop when validation ≥99%

## Structure

```
├── model.py          # CNN
├── data_loader.py    # MNIST loaders & transforms
├── train.py          # Train/validate/test, plots, save
├── requirements.txt
└── README.md
```

## Requirements

- Python ≥3.8
- PyTorch, torchvision, matplotlib, numpy, tqdm