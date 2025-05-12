# Fashion MNIST Classifier with TensorFlow/Keras
## 📌 Project Overview
This script implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the Fashion MNIST dataset. Developed by Michael Doba, the solution demonstrates a complete machine learning workflow from data preparation to model evaluation.

## 🧠 Model Architecture
### The CNN features:

3 Convolutional Layers (32 → 64 → 64 filters) with ReLU activation

MaxPooling Layers for dimensionality reduction

Dropout (0.5) for regularization

Dense Layers (64 units → 10-class softmax output)

## 🛠️ Key Features
Reproducible Training: Fixed random seeds (42) for consistent results

Automated Model Saving: Best model checkpointing

Training Monitoring: Early stopping based on validation loss

### Comprehensive Outputs:

 - Model summary (text file)

 - Training history plots (accuracy/loss)

 - Sample predictions visualization

## 📊 Performance
Achieves ~90% test accuracy after 7 epochs

Efficient training on CPU (~2-3 minutes)

## 🚀 Usage
### Prerequisites:

Make sure you have Python installed

Install required packages:

bash
pip install tensorflow numpy matplotlib
## 📂 Code Structure
The script follows a clear workflow:

### Environment Setup 
- Configures TensorFlow and random seeds

### Data Preparation 
- Loads and preprocesses Fashion MNIST

### Model Definition 
- Builds the CNN architecture

### Training 
- Implements callbacks and validation

### Evaluation 
- Tests and visualizes results
