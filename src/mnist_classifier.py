"""
MNIST Handwritten Digit Classification - Complete CNN Implementation
With model saving, loading, and accessibility features
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
import random

def load_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image dimensions: {x_train.shape[1]}x{x_train.shape[2]}")
    
    return (x_train, y_train), (x_test, y_test)

def show_data_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Distribution of Digits in Training Set', fontsize=14, fontweight='bold')
    plt.xlabel('Digit')
    plt.ylabel('Number of Samples')
    plt.xticks(unique)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom')
    
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/cnn_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_sample_digits(x_train, y_train):
    plt.figure(figsize=(12, 8))
    
    for i in range(20):
        plt.subplot(4, 5, i+1)
        random_idx = random.randint(0, len(x_train)-1)
        plt.imshow(x_train[random_idx], cmap='gray')
        plt.title(f'Label: {y_train[random_idx]}', fontsize=12)
        plt.axis('off')
    
    plt.suptitle('Random Sample Images from Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/cnn_sample_digits.png', dpi=300, bbox_inches='tight')
    plt.show()

def preprocess_data(x_train, y_train, x_test, y_test):
    print("Preprocessing data...")
    
    # Normalize pixels to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Data shape after preprocessing: {x_train.shape}")
    return x_train, y_train, x_test, y_test

def build_cnn_model():
    model = Sequential()
    
    # First conv block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Second conv block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Third conv block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("CNN Model Architecture:")
    model.summary()
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    print("Starting training...")
    
    os.makedirs('../models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint('../models/mnist_cnn_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(x_train, y_train,
                       batch_size=128,
                       epochs=15,
                       validation_data=(x_test, y_test),
                       callbacks=callbacks,
                       verbose=1)
    
    return history

def save_complete_model(model, history, model_name="mnist_cnn_final"):
    """Save model, history, and metadata"""
    print(f"\n=== Saving Complete CNN Model ===")
    
    os.makedirs('../models', exist_ok=True)
    
    # Save the complete model
    model_path = f'../models/{model_name}.h5'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = f'../models/{model_name}_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_path}")
    
    # Save model summary and metadata
    summary_path = f'../models/{model_name}_info.txt'
    with open(summary_path, 'w') as f:
        f.write("MNIST CNN Model Information\n")
        f.write("==========================\n\n")
        f.write("Model Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Total Parameters: {model.count_params()}\n")
    print(f"Model info saved to: {summary_path}")

def load_complete_model(model_name="mnist_cnn_final"):
    """Load saved model and history"""
    model_path = f'../models/{model_name}.h5'
    history_path = f'../models/{model_name}_history.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    history = None
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        print(f"Training history loaded")
    
    return model, history

def plot_training_results(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/cnn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test):
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    return test_accuracy

def show_predictions(model, x_test, y_test, num_samples=12):
    predictions = model.predict(x_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(3, 4, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        
        is_correct = predicted_classes[i] == true_classes[i]
        color = 'green' if is_correct else 'red'
        confidence = predictions[i][predicted_classes[i]] * 100
        
        plt.title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}\n'
                 f'Confidence: {confidence:.1f}%', color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('CNN Model Predictions (Green=Correct, Red=Wrong)', fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/cnn_predictions_sample.png', dpi=300, bbox_inches='tight')
    plt.show()

def quick_test_cnn(model_name="mnist_cnn_final"):
    """Quick test function for saved CNN model"""
    print("=== Quick CNN Model Test ===")
    
    # Load the model
    model, history = load_complete_model(model_name)
    if model is None:
        print("No saved model found. Please train the model first.")
        return None
    
    # Load test data
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test_categorical = to_categorical(y_test, 10)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"Loaded model accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Show some predictions
    predictions = model.predict(x_test[:5])
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nSample predictions:")
    for i in range(5):
        confidence = predictions[i][predicted_classes[i]] * 100
        print(f"True: {y_test[i]}, Predicted: {predicted_classes[i]}, "
              f"Confidence: {confidence:.1f}%")
    
    return model

def interactive_cnn_test():
    """Interactive testing for CNN"""
    print("=== Interactive CNN Testing ===")
    
    model, _ = load_complete_model()
    if model is None:
        print("No saved CNN model found.")
        return
    
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test_processed = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    while True:
        print("\nOptions:")
        print("1. Test random sample")
        print("2. Test specific index")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '3':
            break
        elif choice == '1':
            idx = random.randint(0, len(x_test)-1)
        elif choice == '2':
            try:
                idx = int(input(f"Enter index (0-{len(x_test)-1}): "))
                if not 0 <= idx < len(x_test):
                    print("Invalid index")
                    continue
            except ValueError:
                print("Please enter a valid number")
                continue
        else:
            print("Invalid choice")
            continue
        
        # Make prediction
        sample = x_test_processed[idx:idx+1]
        prediction = model.predict(sample, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100
        true_class = y_test[idx]
        
        # Show image and result
        plt.figure(figsize=(6, 4))
        plt.imshow(x_test[idx], cmap='gray')
        color = 'green' if predicted_class == true_class else 'red'
        plt.title(f'True: {true_class}, Predicted: {predicted_class}\n'
                 f'Confidence: {confidence:.1f}%', color=color, fontsize=14)
        plt.axis('off')
        plt.show()

def main():
    print("=" * 60)
    print("MNIST DIGIT CLASSIFICATION - CNN APPROACH")
    print("=" * 60)
    
    # Check if model already exists
    if os.path.exists('../models/mnist_cnn_final.h5'):
        print("Found existing trained model!")
        choice = input("Do you want to (1) Use existing model or (2) Train new model? Enter 1 or 2: ")
        
        if choice == '1':
            print("Using existing model...")
            quick_test_cnn()
            interactive_cnn_test()
            return
    
    # Train new model
    print("Training new CNN model...")
    
    # Load and explore data
    (x_train, y_train), (x_test, y_test) = load_data()
    show_data_distribution(y_train)
    show_sample_digits(x_train, y_train)
    
    # Preprocess data
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Build and train model
    model = build_cnn_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Show results
    plot_training_results(history)
    final_accuracy = evaluate_model(model, x_test, y_test)
    show_predictions(model, x_test, y_test)
    
    # Save complete model
    save_complete_model(model, history)
    
    # Interactive testing
    interactive_cnn_test()
    
    print(f"\nCNN Training completed! Final accuracy: {final_accuracy*100:.2f}%")
    print("Model saved and ready for future use!")

if __name__ == "__main__":
    main()
