import numpy as np
import matplotlib.pyplot as plt

def load_iris_data(path):
    data_list = []
    label_map = {"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 2.0}
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split(",")  # Split by comma
            features = list(map(float, parts[:4]))  # Convert first 4 values to float
            label = label_map[parts[4]]  # Convert label to float
            data_list.append(features + [label])

    data = np.array(data_list, dtype=float).T  # Transpose to (4, 150) for features and (1, 150) for labels
    return data

def plot_histograms(data):
    features = data[:4, :]  # (4 × 150) -> 4 rows (features), 150 columns (samples)
    labels = data[4, :]

    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    colors = ["red", "blue", "green"]
    class_names = ["Setosa", "Versicolor", "Virginica"]

    # Create histograms for each feature
    plt.figure(figsize=(16, 6))

    for i in range(4):  # Iterate over the 4 features
        plt.subplot(1, 4, i+1)  # 2 rows, 2 columns grid
        for j, class_name in enumerate(class_names):
            plt.hist(features[i, labels == j], bins=8, alpha=0.6, color=colors[j], label=class_name)

        plt.title(f"Histogram of {feature_names[i]}")
        plt.xlabel(feature_names[i])
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()  # Adjust spacing
    plt.show()

def plot_pairs(data):
    features = data[:4, :]  # (4 × 150) -> 4 rows (features), 150 columns (samples)
    labels = data[4, :]  # (1 × 150) -> 150 labels

    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    colors = ["red", "blue", "green"]
    class_names = ["Setosa", "Versicolor", "Virginica"]

    # Create scatter plots for each pair of features
    plt.figure(figsize=(12, 10))

    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)  # Create a 4x4 grid of subplots
            
            if i == j:  # Diagonal: Density plot
                for k, class_name in enumerate(class_names):
                    plt.hist(features[i, labels == k], bins=20, color=colors[k], label=class_name, alpha=0.5, density=True)
                plt.title(f"Density of {feature_names[i]}")
                plt.xlabel(feature_names[i])
                plt.ylabel("Density")
                plt.legend()
            elif i < j:  # Upper triangle: Scatter plot (i vs j)
                for k, class_name in enumerate(class_names):
                    plt.scatter(features[i, labels == k], features[j, labels == k], 
                                color=colors[k], label=class_name, alpha=0.6)
                plt.title(f"Scatter of {feature_names[i]} vs {feature_names[j]}")
                plt.xlabel(feature_names[i])
                plt.ylabel(feature_names[j])
                plt.legend()
            else:  # Lower triangle: Scatter plot (i vs j) reversed (j vs i)
                for k, class_name in enumerate(class_names):
                    plt.scatter(features[i, labels == k], features[j, labels == k], 
                                color=colors[k], label=class_name, alpha=0.6)
                plt.title(f"Scatter of {feature_names[i]} vs {feature_names[j]}")
                plt.xlabel(feature_names[i])
                plt.ylabel(feature_names[j])
                plt.legend()

    plt.tight_layout()
    plt.show()

def mean(features, labels, num_classes=3):
    # Compute the mean of the features for each class
    class_means = []
    for i in range(num_classes):
        class_data = features[:, labels == i]
        class_mean = class_data.mean(axis=1).reshape(features.shape[0], 1)
        class_means.append(class_mean)
    return class_means

def covariance(features, labels, num_classes=3):
    # Compute the covariance matrix for each class
    class_covariances = []
    for i in range(num_classes):
        class_data = features[:, labels == i]
        class_cov = np.cov(class_data, rowvar=True)
        class_covariances.append(class_cov)
    return class_covariances

def variance(features):
    # Calculate the variance for each feature
    return features.var(axis=1)

def std_dev(features):
    # Calculate the standard deviation for each feature
    return features.std(axis=1)
def normalize(features, class_means, labels):
    # Normalize the features by subtracting the class mean for each feature
    normalized_features = np.copy(features)
    for i, class_mean in enumerate(class_means):
        normalized_features[:, labels == i] = features[:, labels == i] - class_mean
    return normalized_features
# Example usage:
data = load_iris_data("/Users/rasmusvikstrom/Desktop/ml_pattern/lab_2/iris.csv")

# Plotting
# plot_histograms(data)
# plot_pairs(data)

# Normalize the data by class
features = data[:4, :]
labels = data[4, :]
class_means = mean(features, labels)
normalized_features = normalize(features, class_means, labels)

# Print Mean, Covariance, Variance, and Standard Deviation for the entire dataset
print("Mean:")
print(np.mean(features, axis=1).reshape(-1, 1))

print("\nCovariance matrix:")
print(np.cov(features, rowvar=True))

print("\nVariance:", variance(features))
print("Std. dev.:", std_dev(features))

# Now print per-class information:
class_names = ["Setosa", "Versicolor", "Virginica"]
for i in range(3):
    print(f"\nClass {i}")
    print("Mean:")
    print(class_means[i])
    print("Covariance:")
    print(covariance(features, labels)[i])
    print("Variance:", variance(features[:, labels == i]))
    print("Std. dev.:", std_dev(features[:, labels == i]))
