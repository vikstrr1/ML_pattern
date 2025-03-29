import numpy as np
import matplotlib.pyplot as plt

def load_dataset(filepath):
    """
    Load the dataset from a text file.
    """
    data = np.loadtxt(filepath, delimiter=',')
    X = data[:, :-1]  # Features
    y = data[:, -1].astype(int)  # Class labels
    return X, y

def compute_ml_estimates(X):
    """
    Compute the Maximum Likelihood estimates for the mean and variance.
    """
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2

def gaussian_density(x, mu, sigma2):
    """
    Compute the Gaussian density for a given x, mean (mu), and variance (sigma2).
    """
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma2)

def plot_feature_density(X, feature_idx, mu, sigma2, class_label):
    """
    Plot the normalized histogram and Gaussian density for a specific feature.
    """
    plt.hist(X[:, feature_idx], bins=30, density=True, alpha=0.6, label="Histogram")
    x_vals = np.linspace(np.min(X[:, feature_idx]), np.max(X[:, feature_idx]), 1000)
    plt.plot(x_vals, gaussian_density(x_vals, mu, sigma2), label="Gaussian Density")
    plt.title(f"Feature {feature_idx + 1} - Class {class_label}")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load the dataset
    filepath = "/Users/rasmusvikstrom/Desktop/ml_pattern/project/trainData.txt"
    X, y = load_dataset(filepath)

    # Split the dataset by class
    classes = np.unique(y)
    for class_label in classes:
        X_class = X[y == class_label]

        # Fit uni-variate Gaussian models for each feature
        mu, sigma2 = compute_ml_estimates(X_class)

        # Plot the density for each feature
        for feature_idx in range(X.shape[1]):
            print(f"Class {class_label}, Feature {feature_idx + 1}: Mean = {mu[feature_idx]:.4f}, Variance = {sigma2[feature_idx]:.4f}")
            plot_feature_density(X_class, feature_idx, mu[feature_idx], sigma2[feature_idx], class_label)