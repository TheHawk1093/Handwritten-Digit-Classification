"""
Complete Model Manager for MNIST Classification Project
Easy access to both CNN and PCA models
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random

def list_available_models():
    """List all available saved models"""
    print("=" * 50)
    print("AVAILABLE SAVED MODELS")
    print("=" * 50)
    
    models_dir = '../models'
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    files = os.listdir(models_dir)
    
    # CNN Models
    cnn_models = [f for f in files if f.endswith('.h5')]
    print(f"\nCNN Models ({len(cnn_models)}):")
    if cnn_models:
        for model in cnn_models:
            size = os.path.getsize(os.path.join(models_dir, model)) / (1024*1024)
            print(f"  - {model} ({size:.1f} MB)")
    else:
        print("  No CNN models found")
    
    # PCA Analyses
    pca_models = [f for f in files if 'pca.pkl' in f]
    print(f"\nPCA Analyses ({len(pca_models)}):")
    if pca_models:
        for model in pca_models:
            analysis_name = model.replace('_pca.pkl', '')
            print(f"  - {analysis_name}")
    else:
        print("  No PCA analyses found")

def quick_comparison():
    """Compare CNN vs PCA performance quickly"""
    print("=" * 50)
    print("QUICK MODEL COMPARISON")
    print("=" * 50)
    
    # Test CNN
    cnn_available = False
    try:
        sys.path.append('../src')
        from mnist_classifier import quick_test_cnn
        print("\nCNN Performance:")
        cnn_model = quick_test_cnn()
        if cnn_model is not None:
            cnn_available = True
    except Exception as e:
        print("CNN model not available")
        print(f"   Error: {str(e)}")
    
    # Test PCA
    pca_available = False
    try:
        from mnist_classifier_pca import quick_test_pca
        print("\nPCA Performance:")
        pca_result = quick_test_pca()
        if pca_result[0] is not None:
            pca_available = True
    except Exception as e:
        print("PCA analysis not available")
        print(f"   Error: {str(e)}")
    
    # Comparison summary
    if cnn_available and pca_available:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print("CNN Approach:")
        print("   + Higher accuracy (~98-99%)")
        print("   + Automatic feature learning")
        print("   - Slower training (~5-10 minutes)")
        print("   - Less interpretable")
        print("   - Larger model size")
        
        print("\nPCA Approach:")
        print("   + Faster training (~1-2 minutes)")
        print("   + More interpretable")
        print("   + Smaller model size")
        print("   + Good dimensionality reduction")
        print("   - Lower accuracy (~91-94%)")
        print("   - Manual feature engineering")

def interactive_predictor():
    """Interactive prediction tool for both models"""
    print("=" * 50)
    print("INTERACTIVE MNIST PREDICTOR")
    print("=" * 50)
    
    # Load test data once
    try:
        (_, _), (x_test, y_test) = mnist.load_data()
        print(f"Loaded {len(x_test)} test samples")
    except Exception as e:
        print(f"Failed to load MNIST data: {e}")
        return
    
    # Check available models
    cnn_available = os.path.exists('../models/mnist_cnn_final.h5')
    pca_available = os.path.exists('../models/mnist_pca_complete_pca.pkl')
    
    print(f"\nAvailable models:")
    print(f"  CNN: {'Available' if cnn_available else 'Not Available'}")
    print(f"  PCA: {'Available' if pca_available else 'Not Available'}")
    
    if not (cnn_available or pca_available):
        print("\nNo trained models found. Please train models first.")
        return
    
    while True:
        print("\n" + "-" * 30)
        print("PREDICTION OPTIONS")
        print("-" * 30)
        print("1. Test CNN model on random sample")
        print("2. Test PCA model on random sample")
        print("3. Compare both models on same sample")
        print("4. Test on specific sample index")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '5':
            print("Goodbye!")
            break
        
        # Get sample
        if choice == '4':
            try:
                idx = int(input(f"Enter sample index (0-{len(x_test)-1}): "))
                if not 0 <= idx < len(x_test):
                    print("Invalid index")
                    continue
            except ValueError:
                print("Please enter a valid number")
                continue
        else:
            idx = random.randint(0, len(x_test)-1)
        
        sample_image = x_test[idx]
        true_label = y_test[idx]
        
        # Show the image
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_image, cmap='gray')
        plt.title(f'Sample #{idx}\nTrue Label: {true_label}', fontsize=14)
        plt.axis('off')
        
        # Predictions subplot
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        predictions_text = f"True Label: {true_label}\n\n"
        
        # Test CNN
        if choice in ['1', '3'] and cnn_available:
            try:
                sys.path.append('../src')
                from mnist_classifier import load_complete_model
                model, _ = load_complete_model()
                if model:
                    sample_processed = sample_image.reshape(1, 28, 28, 1).astype('float32') / 255.0
                    prediction = model.predict(sample_processed, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class] * 100
                    
                    status = 'CORRECT' if predicted_class == true_label else 'WRONG'
                    predictions_text += f"[{status}] CNN Prediction: {predicted_class}\n"
                    predictions_text += f"   Confidence: {confidence:.1f}%\n\n"
            except Exception as e:
                predictions_text += f"CNN Error: {str(e)}\n\n"
        
        # Test PCA
        if choice in ['2', '3'] and pca_available:
            try:
                from mnist_classifier_pca import load_pca_analysis
                pca, scaler, classifiers, results = load_pca_analysis()
                if pca and classifiers:
                    sample_flat = sample_image.reshape(1, -1).astype('float32') / 255.0
                    sample_scaled = scaler.transform(sample_flat)
                    sample_pca = pca.transform(sample_scaled)
                    
                    # Use best classifier
                    best_name = max(classifiers.keys(), 
                                  key=lambda k: results[k]['accuracy'] if results else 0.5)
                    best_classifier = classifiers[best_name]
                    prediction = best_classifier.predict(sample_pca)[0]
                    
                    status = 'CORRECT' if prediction == true_label else 'WRONG'
                    predictions_text += f"[{status}] PCA Prediction: {prediction}\n"
                    predictions_text += f"   Classifier: {best_name}\n"
            except Exception as e:
                predictions_text += f"PCA Error: {str(e)}\n"
        
        plt.text(0.1, 0.5, predictions_text, fontsize=12, verticalalignment='center',
                transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()

def model_info():
    """Show detailed information about saved models"""
    print("=" * 50)
    print("DETAILED MODEL INFORMATION")
    print("=" * 50)
    
    models_dir = '../models'
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    # CNN Model Info
    cnn_info_file = '../models/mnist_cnn_final_info.txt'
    if os.path.exists(cnn_info_file):
        print("\nCNN Model Information:")
        print("-" * 30)
        with open(cnn_info_file, 'r') as f:
            print(f.read())
    
    # PCA Analysis Info
    pca_info_file = '../models/mnist_pca_complete_summary.txt'
    if os.path.exists(pca_info_file):
        print("\nPCA Analysis Information:")
        print("-" * 30)
        with open(pca_info_file, 'r') as f:
            print(f.read())

def main():
    print("MNIST MODEL MANAGER")
    print("=" * 50)
    print("Welcome to the MNIST Classification Model Manager!")
    print("This tool helps you access and compare your trained models.")
    
    while True:
        print("\n" + "=" * 30)
        print("MAIN MENU")
        print("=" * 30)
        print("1. List available models")
        print("2. Quick model comparison")
        print("3. Interactive predictor")
        print("4. Detailed model information")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        try:
            if choice == '1':
                list_available_models()
            elif choice == '2':
                quick_comparison()
            elif choice == '3':
                interactive_predictor()
            elif choice == '4':
                model_info()
            elif choice == '5':
                print("\nThank you for using MNIST Model Manager!")
                print("Happy machine learning!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or report this issue.")

if __name__ == "__main__":
    main()

