# Fresh-or-Rotten-Fruit-Classification-Assessment

This NVIDIA project implements a deep learning model to classify images of fresh and rotten fruits. The primary goal is to achieve at least 92% validation accuracy using transfer learning, data augmentation, and fine-tuning techniques.

## 📋 Project Overview
The model is trained to recognize six distinct categories of fruit:

- Fresh Apples
- Fresh Oranges
- Fresh Bananas
- Rotten Apples
- Rotten Oranges
- Rotten Bananas

## 🛠️ Technical Stack
- **Framework**: PyTorch
- **Base Model**: VGG16 (Pre-trained on ImageNet)
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Augmentation**: TorchVision Transforms (Random Resized Crop, Horizontal Flip)

## 📂 File Structure

### 1. `01_assessment.ipynb`
The main notebook containing the training pipeline:
- **Model Architecture**: Combines VGG16's feature extractor with custom linear layers (4096 → 500 → 6).
- **Transfer Learning**: Initially freezes the base VGG16 weights to train only the custom classifier.
- **Fine-Tuning**: Unfreezes the base model for a final training pass with a low learning rate ($1 \times 10^{-4}$).
- **Performance**: Achieves an accuracy of approximately 96.6%, exceeding the 92% requirement.

### 2. `utils.py`
A helper module containing essential functions for the training loop:
- **MyConvBlock**: A modular convolutional block with BatchNorm and Dropout.
- **get_batch_accuracy**: Calculates accuracy for a specific batch.
- **train**: Handles the training loop, including backpropagation and optimizer steps.
- **validate**: Evaluates the model on the validation set without updating gradients.

## 🚀 Getting Started

### Data Loading
The project uses a custom `MyDataset` class to load `.png` images from the `data/fruits/` directory. Images are preprocessed using the standard VGG16 normalization transforms.

### Training Process
1. **Initial Training**: Train the head of the model for 10 epochs while the VGG base is frozen.
2. **Fine-Tuning**: Unfreeze the entire network and train for an additional epoch with a reduced learning rate to refine the weights for the specific fruit dataset.

### Execution
To run the training and evaluation:
1. Ensure the `data/` folder is populated with the Kaggle fruits dataset.
2. Execute the cells in `01_assessment.ipynb` sequentially.
3. Use the `run_assessment(my_model)` function to verify if the model meets the passing criteria.

## 📈 Results
The provided logs show the model successfully reaching a validation accuracy of **96.66%** after 10 epochs of initial training and 1 epoch of fine-tuning.
