import numpy as np
import matplotlib.pyplot as plt

def logpdf_GAU_ND(X, mu, C):
    """
    Compute the log-density of a Multivariate Gaussian distribution.
    
    Parameters:
        X (numpy.ndarray): Data matrix of shape (M, N), where M is the feature size and N is the number of samples.
        mu (numpy.ndarray): Mean vector of shape (M, 1).
        C (numpy.ndarray): Covariance matrix of shape (M, M).
    
    Returns:
        numpy.ndarray: Log-density values for each sample, shape (N,).
    """
    M = X.shape[0]
    XC = X - mu
    invC = np.linalg.inv(C)
    logdetC = np.linalg.slogdet(C)[1]
    const = -0.5 * M * np.log(2 * np.pi)
    log_density = const - 0.5 * logdetC - 0.5 * np.sum(XC * (invC @ XC), axis=0)
    return log_density

def ML_estimate(X):
    """
    Compute the Maximum Likelihood estimates for the mean and covariance matrix.
    
    Parameters:
        X (numpy.ndarray): Data matrix of shape (M, N), where M is the feature size and N is the number of samples.
    
    Returns:
        tuple: Mean vector (numpy.ndarray of shape (M, 1)) and covariance matrix (numpy.ndarray of shape (M, M)).
    """
    mu_ML = np.mean(X, axis=1, keepdims=True)
    XC = X - mu_ML
    C_ML = (XC @ XC.T) / X.shape[1]
    return mu_ML, C_ML

def log_likelihood(X, mu, C):
    """
    Compute the log-likelihood of the data given the Gaussian parameters.
    
    Parameters:
        X (numpy.ndarray): Data matrix of shape (M, N), where M is the feature size and N is the number of samples.
        mu (numpy.ndarray): Mean vector of shape (M, 1).
        C (numpy.ndarray): Covariance matrix of shape (M, M).
    
    Returns:
        float: Log-likelihood value.
    """
    log_densities = logpdf_GAU_ND(X, mu, C)
    return np.sum(log_densities)

if __name__ == "__main__":
    # Example usage for 1D Gaussian
    X1D = np.load('/Users/rasmusvikstrom/Desktop/ml_pattern/lab_5/solution/X1D.npy')
    m_ML, C_ML = ML_estimate(X1D)
    print("ML Estimates for 1D Gaussian:")
    print("Mean:", m_ML)
    print("Covariance:", C_ML)

    ll = log_likelihood(X1D, m_ML, C_ML)
    print("Log-likelihood:", ll)

    # Plot histogram and density
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True, alpha=0.6, label="Histogram")
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, -1), m_ML, C_ML)), label="Density")
    plt.legend()
    plt.show()

    # Example usage for ND Gaussian
    XND = np.load('/Users/rasmusvikstrom/Desktop/ml_pattern/lab_5/solution/XND.npy')
    mu_ML, C_ML = ML_estimate(XND)
    print("ML Estimates for ND Gaussian:")
    print("Mean:", mu_ML)
    print("Covariance:", C_ML)

    ll = log_likelihood(XND, mu_ML, C_ML)
    print("Log-likelihood:", ll)