"""
MNIST Digit Classification using PCA + Machine Learning - Complete Implementation
With model saving, loading, and accessibility features
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
import seaborn as sns
import os
import time
import joblib
import pickle

def load_and_prepare_data():
    """Load MNIST and convert from 2D images to 1D feature vectors"""
    print("=== Loading MNIST Dataset ===")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Convert 28x28 images to 784-length vectors
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Normalize pixel values to 0-1 range
    x_train_flat = x_train_flat.astype('float32') / 255.0
    x_test_flat = x_test_flat.astype('float32') / 255.0
    
    print(f"Training samples: {x_train_flat.shape[0]}")
    print(f"Test samples: {x_test_flat.shape[0]}")
    print(f"Features per image: {x_train_flat.shape[1]} (28x28 = 784)")
    
    return x_train_flat, y_train, x_test_flat, y_test, x_train, x_test

def show_original_samples(x_train, y_train):
    """Display sample digits before any processing"""
    plt.figure(figsize=(12, 6))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f'Digit: {y_train[i]}')
        plt.axis('off')
    
    plt.suptitle('Original MNIST Samples', fontsize=16)
    plt.tight_layout()
    
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/pca_original_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_variance_requirements(x_train_flat):
    """Find out how many components we need for different variance levels"""
    print("\n=== Analyzing PCA Variance Requirements ===")
    
    # Standardize data first (important for PCA)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_flat)
    
    # Test different variance retention levels
    variance_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    components_needed = []
    
    for variance in variance_levels:
        pca_temp = PCA(n_components=variance)
        pca_temp.fit(x_scaled)
        components_needed.append(pca_temp.n_components_)
        
        compression_ratio = 784 / pca_temp.n_components_
        print(f"For {variance*100:2.0f}% variance: {pca_temp.n_components_:3d} components "
              f"(compression: {compression_ratio:.1f}x)")
    
    # Visualize the trade-off
    plt.figure(figsize=(10, 6))
    plt.plot([v*100 for v in variance_levels], components_needed, 
             'bo-', linewidth=3, markersize=8)
    plt.xlabel('Variance Retained (%)', fontsize=12)
    plt.ylabel('Components Required', fontsize=12)
    plt.title('PCA: Variance vs Components Trade-off', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for i, (var, comp) in enumerate(zip(variance_levels, components_needed)):
        plt.annotate(f'{comp}', (var*100, comp), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig('../results/pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scaler

def apply_pca_reduction(x_train_flat, x_test_flat, scaler, variance_keep=0.95):
    """Apply PCA to reduce dimensionality while keeping specified variance"""
    print(f"\n=== Applying PCA (keeping {variance_keep*100}% variance) ===")
    
    # Standardize the data
    x_train_scaled = scaler.transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    
    # Apply PCA
    pca = PCA(n_components=variance_keep)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    
    print(f"Original dimensions: {x_train_flat.shape[1]}")
    print(f"Reduced dimensions: {x_train_pca.shape[1]}")
    print(f"Compression ratio: {x_train_flat.shape[1]/x_train_pca.shape[1]:.1f}x")
    print(f"Actual variance retained: {pca.explained_variance_ratio_.sum():.3f}")
    
    return x_train_pca, x_test_pca, pca

def visualize_eigendigits(pca):
    """Show what the principal components look like as images"""
    print("\n=== Visualizing Principal Components (Eigendigits) ===")
    
    plt.figure(figsize=(15, 8))
    n_components_to_show = min(20, pca.n_components_)
    
    for i in range(n_components_to_show):
        plt.subplot(4, 5, i+1)
        # Reshape component back to 28x28 image
        eigendigit = pca.components_[i].reshape(28, 28)
        plt.imshow(eigendigit, cmap='RdBu_r')
        plt.title(f'PC{i+1}\n{pca.explained_variance_ratio_[i]:.3f}')
        plt.axis('off')
    
    plt.suptitle('Principal Components (Eigendigits)\nThese are the building blocks PCA uses', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/pca_eigendigits.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_2d_visualization(x_train_pca, y_train):
    """Project data to 2D space to see how different digits cluster"""
    print("\n=== Creating 2D Visualization ===")
    
    # Use only first 2 components for visualization
    pca_2d = PCA(n_components=2)
    x_2d = pca_2d.fit_transform(x_train_pca[:5000])  # Use subset for speed
    y_subset = y_train[:5000]
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = y_subset == digit
        plt.scatter(x_2d[mask, 0], x_2d[mask, 1], 
                   c=[colors[digit]], label=f'Digit {digit}', 
                   alpha=0.6, s=15)
    
    plt.xlabel(f'1st Principal Component')
    plt.ylabel(f'2nd Principal Component')
    plt.title('MNIST Digits in 2D PCA Space\nSee how digits naturally cluster!')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/pca_2d_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_multiple_classifiers(x_train_pca, y_train, x_test_pca, y_test):
    """Train different classifiers on PCA-reduced data and compare performance"""
    print("\n=== Training Multiple Classifiers ===")
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, gamma='scale'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Time the training
        start_time = time.time()
        classifier.fit(x_train_pca, y_train)
        train_time = time.time() - start_time
        
        # Time the prediction
        start_time = time.time()
        y_pred = classifier.predict(x_test_pca)
        predict_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'classifier': classifier,
            'accuracy': accuracy,
            'predictions': y_pred,
            'train_time': train_time,
            'predict_time': predict_time
        }
        
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Training time: {train_time:.2f} seconds")
        print(f"  Prediction time: {predict_time:.2f} seconds")
    
    return results

def save_pca_analysis(pca, scaler, results, analysis_name="mnist_pca_complete"):
    """Save complete PCA analysis"""
    print(f"\n=== Saving Complete PCA Analysis ===")
    
    os.makedirs('../models', exist_ok=True)
    
    # Save PCA object
    pca_path = f'../models/{analysis_name}_pca.pkl'
    joblib.dump(pca, pca_path)
    print(f"PCA object saved to: {pca_path}")
    
    # Save scaler
    scaler_path = f'../models/{analysis_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save all trained classifiers
    classifiers_path = f'../models/{analysis_name}_classifiers.pkl'
    classifier_dict = {name: result['classifier'] for name, result in results.items()}
    joblib.dump(classifier_dict, classifiers_path)
    print(f"All classifiers saved to: {classifiers_path}")
    
    # Save complete results
    results_path = f'../models/{analysis_name}_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Complete results saved to: {results_path}")
    
    # Save analysis summary
    summary_path = f'../models/{analysis_name}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PCA Analysis Summary\n")
        f.write("==================\n\n")
        f.write(f"Original features: 784\n")
        f.write(f"PCA features: {pca.n_components_}\n")
        f.write(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}\n")
        f.write(f"Compression ratio: {784/pca.n_components_:.1f}x\n\n")
        f.write(f"Classifier Results:\n")
        f.write("-" * 40 + "\n")
        for name, result in results.items():
            f.write(f"{name:<20}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
        
        best_classifier = max(results.keys(), key=lambda k: results[k]['accuracy'])
        f.write(f"\nBest Classifier: {best_classifier}\n")
        f.write(f"Best Accuracy: {results[best_classifier]['accuracy']:.4f}\n")
    
    print(f"Analysis summary saved to: {summary_path}")

def load_pca_analysis(analysis_name="mnist_pca_complete"):
    """Load complete saved PCA analysis"""
    pca_path = f'../models/{analysis_name}_pca.pkl'
    scaler_path = f'../models/{analysis_name}_scaler.pkl'
    classifiers_path = f'../models/{analysis_name}_classifiers.pkl'
    results_path = f'../models/{analysis_name}_results.pkl'
    
    if not all(os.path.exists(path) for path in [pca_path, scaler_path, classifiers_path]):
        print("PCA analysis files not found. Please run the analysis first.")
        return None, None, None, None
    
    print(f"Loading complete PCA analysis...")
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)
    classifiers = joblib.load(classifiers_path)
    
    results = None
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    
    print(f"Loaded PCA with {pca.n_components_} components")
    print(f"Loaded {len(classifiers)} classifiers")
    
    return pca, scaler, classifiers, results

def quick_test_pca(analysis_name="mnist_pca_complete"):
    """Quick test function for saved PCA analysis"""
    print("=== Quick PCA Analysis Test ===")
    
    # Load saved analysis
    pca, scaler, classifiers, results = load_pca_analysis(analysis_name)
    if pca is None:
        return None
    
    # Load test data
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test_flat = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Transform test data
    x_test_scaled = scaler.transform(x_test_flat[:1000])  # Test on first 1000 samples
    x_test_pca = pca.transform(x_test_scaled)
    
    print(f"Test data transformed: {x_test_flat[:1000].shape} -> {x_test_pca.shape}")
    
    # Test each classifier
    for name, classifier in classifiers.items():
        y_pred = classifier.predict(x_test_pca)
        accuracy = accuracy_score(y_test[:1000], y_pred)
        print(f"{name:<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return pca, scaler, classifiers, results

def compare_classifiers(results):
    """Create comparison charts for different classifiers"""
    plt.figure(figsize=(15, 5))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    train_times = [results[name]['train_time'] for name in names]
    predict_times = [results[name]['predict_time'] for name in names]
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    bars = plt.bar(names, accuracies, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Classification Accuracy', fontsize=14)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    plt.subplot(1, 3, 2)
    plt.bar(names, train_times, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Training Time', fontsize=14)
    plt.ylabel('Seconds')
    plt.xticks(rotation=45)
    
    # Prediction time comparison
    plt.subplot(1, 3, 3)
    plt.bar(names, predict_times, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Prediction Time', fontsize=14)
    plt.ylabel('Seconds')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/pca_classifier_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_detailed_results(y_test, results):
    """Show confusion matrix and detailed results for best classifier"""
    # Find best classifier
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_name]
    
    print(f"\n=== Detailed Results for Best Classifier: {best_name} ===")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {best_name}', fontsize=16)
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.savefig('../results/pca_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, best_result['predictions']))

def interactive_pca_test():
    """Interactive testing for PCA models"""
    print("=== Interactive PCA Testing ===")
    
    pca, scaler, classifiers, results = load_pca_analysis()
    if pca is None:
        print("No saved PCA analysis found.")
        return
    
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test_flat = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Get best classifier
    best_name = max(classifiers.keys(), key=lambda k: results[k]['accuracy'] if results else 0.5)
    best_classifier = classifiers[best_name]
    
    while True:
        print(f"\nUsing {best_name} classifier")
        print("Options:")
        print("1. Test random sample")
        print("2. Test specific index")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '3':
            break
        elif choice == '1':
            idx = np.random.randint(0, len(x_test))
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
        
        # Transform and predict
        sample_flat = x_test_flat[idx:idx+1]
        sample_scaled = scaler.transform(sample_flat)
        sample_pca = pca.transform(sample_scaled)
        prediction = best_classifier.predict(sample_pca)[0]
        true_label = y_test[idx]
        
        # Show image and result
        plt.figure(figsize=(6, 4))
        plt.imshow(x_test[idx], cmap='gray')
        color = 'green' if prediction == true_label else 'red'
        plt.title(f'True: {true_label}, Predicted: {prediction}', 
                 color=color, fontsize=14)
        plt.axis('off')
        plt.show()

def main():
    print("=" * 60)
    print("MNIST DIGIT CLASSIFICATION - PCA APPROACH")
    print("=" * 60)
    
    # Check if analysis already exists
    if os.path.exists('../models/mnist_pca_complete_pca.pkl'):
        print("Found existing PCA analysis!")
        choice = input("Do you want to (1) Use existing analysis or (2) Run new analysis? Enter 1 or 2: ")
        
        if choice == '1':
            print("Using existing PCA analysis...")
            quick_test_pca()
            interactive_pca_test()
            return
    
    # Run new analysis
    print("Running new PCA analysis...")
    
    # Load and prepare data
    x_train_flat, y_train, x_test_flat, y_test, x_train, x_test = load_and_prepare_data()
    
    # Show data characteristics
    show_original_samples(x_train, y_train)
    
    # Analyze variance requirements
    scaler = analyze_variance_requirements(x_train_flat)
    
    # Apply PCA transformation
    x_train_pca, x_test_pca, pca = apply_pca_reduction(x_train_flat, x_test_flat, scaler)
    
    # Visualize principal components
    visualize_eigendigits(pca)
    
    # Create 2D visualization
    create_2d_visualization(x_train_pca, y_train)
    
    # Train multiple classifiers
    results = train_multiple_classifiers(x_train_pca, y_train, x_test_pca, y_test)
    
    # Compare classifier performance
    compare_classifiers(results)
    
    # Show detailed results
    show_detailed_results(y_test, results)
    
    # Save complete analysis
    save_pca_analysis(pca, scaler, results)
    
    # Interactive testing
    interactive_pca_test()
    
    # Summary
    best_classifier = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_classifier]['accuracy']
    
    print("\n" + "=" * 60)
    print("PCA ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Best classifier: {best_classifier}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Dimensionality reduction: {784/pca.n_components_:.1f}x")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.3f}")
    print("All results saved and ready for future use!")

if __name__ == "__main__":
    main()
