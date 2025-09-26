# Handwritten Digit Recognition with CNN

A PyTorch implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model achieves 99%+ validation accuracy using dropout and batch normalization techniques.

## Features

- **Advanced CNN Architecture**: Multi-layer CNN with batch normalization and dropout
- **Data Augmentation**: Random rotation and translation for improved generalization
- **Comprehensive Training**: Training loop with validation, early stopping, and learning rate scheduling
- **Visualization**: Training progress plots and prediction visualizations
- **High Accuracy**: Achieves 99%+ validation accuracy on MNIST dataset

## Project Structure

```
├── model.py           # CNN model architecture with dropout & batch normalization
├── data_loader.py     # MNIST dataset loading and preprocessing
├── train.py          # Main training script with evaluation and visualization
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Model Architecture

The CNN consists of:
- **3 Convolutional Blocks**: Each with conv layers, batch normalization, ReLU activation, max pooling, and dropout
- **3 Fully Connected Layers**: With batch normalization and dropout for regularization
- **Output Layer**: 10 classes for digits 0-9

Key features:
- Batch normalization for stable training
- Dropout (25% for conv layers, 50% for FC layers) to prevent overfitting
- Progressive feature map increase: 32 → 64 → 128 channels

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script:
```bash
python train.py
```

This will:
- Download and preprocess the MNIST dataset
- Train the CNN model
- Display training progress with accuracy and loss metrics
- Generate visualization plots
- Save the best model checkpoint

### Testing Individual Components

Test the model architecture:
```bash
python model.py
```

Test the data loader:
```bash
python data_loader.py
```

## Training Configuration

Default hyperparameters:
- **Batch Size**: 128
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Optimizer**: Adam with weight decay (1e-4)
- **Epochs**: 30 (with early stopping at 99% validation accuracy)
- **Validation Split**: 10% of training data

## Data Augmentation

The training pipeline includes:
- Random rotation (±10 degrees)
- Random translation (±10% in both directions)
- Normalization with MNIST statistics (mean=0.1307, std=0.3081)

## Results

The model typically achieves:
- **Validation Accuracy**: 99%+
- **Test Accuracy**: 99%+
- **Training Time**: ~5-10 minutes on CPU, ~2-3 minutes on GPU

## Output Files

After training, the following files are generated:
- `digit_recognition_cnn.pth`: Best model checkpoint
- `training_history.png`: Training and validation curves
- `predictions.png`: Sample predictions visualization

## Model Performance

The model demonstrates excellent performance across all digit classes with:
- Robust generalization through data augmentation
- Stable training via batch normalization
- Overfitting prevention through dropout
- Efficient learning with Adam optimizer

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib 3.7+
- numpy 1.24+
- tqdm 4.65+

## License

This project is open source and available under the MIT License.