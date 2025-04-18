import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.datasets

def load_iris_data(path):
    data_list = []      # Create an empty list
    label_map = {"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 2.0}  # Create a dictionary              
    with open(path, "r") as file:  # Open the file
        for line in file:  # Iterate over each line
            parts = line.strip().split(",")  # Split by comma
            features = list(map(float, parts[:4]))  # Convert first 4 values to float
            label = label_map[parts[4]]  # Convert label to float
            data_list.append(features + [label])  # Append features and label to the list
        return np.array(data_list, dtype=float).T  
    
def compute_covariance_matrix(data):
    """ Compute the covariance matrix of the dataset. """
    features = data[:4, :]
    mean = features.mean(axis=1)  
    centered_data = features - mean.reshape((mean.size, 1))  
    N = centered_data.shape[1]  
    C = (1 / N) * np.dot(centered_data, centered_data.T)
    return C, centered_data

def compute_pca(C, m=2):
    """ Compute the first m principal components of the covariance matrix C. """
    eigvals, eigvecs = np.linalg.eig(C)
    indices = np.argsort(eigvals)[::-1]  
    P = eigvecs[:, indices[:m]]
    return P
def project_data(P, data):
    """ Project the data onto the space spanned by the principal components P. """
    return np.dot(P.T, data)
def plot_pca_projection(DP,labels):
    """ Plot the PCA-projected data in 2D space. """
    plt.figure(figsize=(8, 6))
    for label, color, name in zip([0.0, 1.0, 2.0], ['red', 'blue', 'green'], 
                                  ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]):
        plt.scatter(DP[0, labels == label], DP[1, labels == label], 
                    c=color, label=name, alpha=0.7, edgecolors="k")
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of IRIS Dataset (2D)")
    plt.legend()
    plt.grid()
    plt.show()

def compute_lda_matrices(data):
    """
    Compute the within-class scatter matrix (Sw) and the between-class scatter matrix (Sb).
    """
    features = data[:4, :]  # Extract feature matrix (4 features, N samples)
    labels = data[4, :].astype(int)  # Extract class labels as integers
    classes = np.unique(labels)  # Unique class labels (0,1,2 for IRIS dataset)
    
    # Compute overall mean vector (global mean of all samples)
    mu = np.mean(features, axis=1, keepdims=True)

    # Initialize scatter matrices
    Sw = np.zeros((features.shape[0], features.shape[0]))  # Within-class scatter matrix
    Sb = np.zeros((features.shape[0], features.shape[0]))  # Between-class scatter matrix

    N = features.shape[1]  # Total number of samples

    for c in classes:
        Dc = features[:, labels == c]  # Select all samples of class c
        nc = Dc.shape[1]  # Number of samples in class c
        mu_c = np.mean(Dc, axis=1, keepdims=True)  # Compute mean of class c

        # Compute within-class scatter (Sw)
        Dc_centered = Dc - mu_c
        Sw += np.dot(Dc_centered, Dc_centered.T)  # Sum up covariance matrices

        # Compute between-class scatter (Sb)
        diff = mu_c - mu
        Sb += nc * np.dot(diff, diff.T)  # Weighted by class size

    return Sw / N, Sb / N

def compute_lda(Sw, Sb, m=2):
    """
    Compute the first m linear discriminants of the LDA transformation.
    """
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]  # Select the first m eigenvectors
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m] 
    return W






data =load_iris_data('/Users/rasmusvikstrom/Desktop/ml_pattern/lab_3/iris.csv')
C, centered_data = compute_covariance_matrix(data)
P = compute_pca(C, m=3)

# Project data onto PCA space
DP = project_data(P, centered_data)
DP[0, :] *= -1  # Flip first PCA axis
DP[1, :] *= -1  # Flip second PCA axis

# Plot the result
#plot_pca_projection(DP, labels=data[4, :])
Sw, Sb = compute_lda_matrices(data)
W =compute_lda(Sw, Sb, m=2)

matrix = np.load('/Users/rasmusvikstrom/Desktop/ml_pattern/lab_3/solution/IRIS_LDA_matrix_m2.npy')

# Print the shape and contents of the matrix to understand its structure
print(matrix.shape)
print(matrix)

print(W)
U, _, Vt = np.linalg.svd(W)

# Perform SVD on the matrix from the file
U_matrix, _, Vt_matrix = np.linalg.svd(matrix)

# Check the similarity of the subspaces
singular_values = np.linalg.svd(np.hstack([U, U_matrix]))[1]

print(singular_values)

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

DIris, LIris = load_iris()
D = DIris[:, LIris != 0]
L = LIris[LIris != 0]

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

# Apply LDA transformation to training and validation data
DTR_lda = np.dot(W.T, DTR)
DVAL_lda = np.dot(W.T, DVAL)

threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0
PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
PVAL[DVAL_lda[0] >= threshold] = 2
PVAL[DVAL_lda[0] < threshold] = 1

# No error rate calculation; just printing LDA transformation results

print('Labels:     ', LVAL)
print('Predictions:', PVAL)
print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
