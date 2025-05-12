# ==================================================================
# Convolutional Neural Network (CNN) built with TensorFlow/Keras to 
# classify images from the Fashion MNIST dataset by Michael Doba
# ==================================================================

# Importing necessary libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models, Input  # type: ignore
import warnings

# Ignore UserWarnings to keep the output clean
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================================================================
#                --- Data Loading and Preparation ---
# ==================================================================

print(f"\nLoading & Preparing Data.................\n")
# Loading Fashion MNIST dataset from Keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class names corresponding to the labels (0-9)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing pixel values to [0, 1] and convert to float32
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include channel dimension (required for Conv2D layers)
x_train = x_train.reshape(-1, 28, 28, 1)  # (samples, height, width, channels)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==================================================================
#                    --- Output Directory Setup ---
# ==================================================================

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create a subdirectory for saving outputs
output_dir = os.path.join(script_dir, "predictions_output")
os.makedirs(output_dir, exist_ok=True)  # Create dir if it doesn't exist

# ==================================================================
#                      --- Model Definition ---
# ==================================================================

# Sequential model with convolutional and dense layers
model = models.Sequential([
    Input(shape=(28, 28, 1)),  # Input layer specifying the shape
    layers.Conv2D(32, (3, 3), activation='relu'),  # 32 filters, 3x3 kernel
    layers.MaxPooling2D((2, 2)),  # Downsampling by taking max over 2x2 blocks
    layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filters
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Another 64-filter conv layer
    layers.Flatten(),  # Flatten 3D output to 1D for dense layers
    layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units
    layers.Dropout(0.5),  # Dropout for regularization (50% dropout rate)
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (using softmax for probabilities)
])

# Compiling the model with optimizer, loss function, and metrics
model.compile(
    optimizer='adam',  # Adaptive Moment Estimation optimizer
    loss='sparse_categorical_crossentropy',  # Loss function for integer labels
    metrics=['accuracy']  # Track accuracy during training
)

# ==================================================================
#                          --- Callbacks ---
# ==================================================================

# EarlyStopping: Stop training if validation loss doesn't improve for 2 epochs
# ModelCheckpoint: Save the best model based on validation loss
callbacks = [
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_model.keras'),
        save_best_only=True
    )
]

# ==================================================================
#                         --- Model Training ---
# ==================================================================

# Train the model on the training data with validation split
history = model.fit(
    x_train,
    y_train,
    epochs=7,  # Number of training iterations
    batch_size=64,  # Number of samples per gradient update
    validation_split=0.1,  # Use 10% of training data for validation
    callbacks=callbacks,  # Apply defined callbacks
    verbose=1  # Show progress bar
)

# ==================================================================
#                         --- Model Evaluation ---
# ==================================================================

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save model summary to a text file
with open(os.path.join(output_dir, "model_summary.txt"), "w", encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# ==================================================================
#                      --- Plot Training History ---
# ==================================================================

plt.figure(figsize=(12, 5))
# Plot accuracy over epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)  # Save plot
plt.close()  # Close the figure to free memory

# ==================================================================
#                       --- Sample Predictions ---
# ==================================================================

# Selecting sample images from the test set
sample_indices = [0, 1, 2]
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]
predictions = model.predict(sample_images, verbose=0)  # Get model predictions

# Plotting predictions alongside true labels
plt.figure(figsize=(10, 5))
for i, (img, true_label, pred) in enumerate(zip(sample_images, sample_labels, predictions)):
    plt.subplot(1, len(sample_indices), i+1)
    plt.imshow(img.squeeze(), cmap='gray')  # Remove channel dim for display
    predicted_label = np.argmax(pred)  # Get the class with highest probability
    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300)  # Save predictions plot
plt.close()

print(f"\nAll outputs saved to: {output_dir}")