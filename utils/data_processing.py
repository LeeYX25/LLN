import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler for tensors

        Param:
            mean: The mean of the features. The property will be set after a call to fit.
            std: The standard deviation of the features. The property will be set after a call to fit.
            epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


def X_to_tensor(X) -> torch.Tensor:
    """Transform input data X to a PyTorch tensor and performs standardization.

    Args:
        X (array-like or pandas.DataFrame): Input data.

    Returns:
        X (torch.Tensor): Standardized input data as a PyTorch tensor.
    """
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype= torch.float)
    elif isinstance(X, torch.Tensor):
        X = X.float()
    else:
        X = torch.tensor(X.astype(float), dtype=torch.float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
        

def y_to_tensor(y) -> torch.Tensor:
    """Convert target data y to a PyTorch tensor.

    Args:
        y (array-like or pandas.DataFrame): Target data.

    Returns:
        y (torch.Tensor): Target data as a PyTorch tensor.
    """ 
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    if isinstance(y, (list, np.ndarray)):
        y = np.array(y).flatten()

    y_tensor = torch.tensor(y, dtype=torch.float)

    return y_tensor

def data_to_tensor(X, y) :
    """Convert input data X and target data y to a PyTorch tensor.

    Args:
        X (array-like or pandas.DataFrame): Input data.
        y (array-like or pandas.DataFrame): Target data.

    Returns:
        X (torch.Tensor): Standardized input data as a PyTorch tensor.
        y (torch.Tensor): Target data as a PyTorch tensor.
    """
    X = X_to_tensor(X)
    y = y_to_tensor(y)
    return X, y


def generate_linear_data(X, s, test_size=None):
    """Generate y values given data X and optionally a test set.
    
    Args:
        X (array-like or pandas.DataFrame): Input data.
        s (int): Number of important features.
        test_size (int, optional): Size of the test set. If None, no test set is created.
        
    Returns:
        y (pandas.DataFrame): Target data
        features (array-like): Feature indices used
        X_test, y_test (pandas.DataFrame, optional): Test sets
    """
    n, p = X.shape
    snr = 10 / np.sqrt(n)
    
    inds = np.random.choice(range(p), s, replace=False)
    beta = snr * np.ones(s) 

    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X

    y = np.dot(X_np[:, inds], beta) + np.random.normal(size=(n))
    y = pd.DataFrame(y)

    if test_size is not None:
        X_test = np.random.normal(size=(test_size, p))
        y_test = np.dot(X_test[:, inds], beta) + np.random.normal(size=(test_size))
        X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
        return y, inds, X_test, y_test
    
    return y, inds


def generate_nonlinear_data(X, s, test_size=None):
    """Generate y values given data X and optionally a test set.
    
    Args:
        X (array-like or pandas.DataFrame): Input data.
        s (int): Number of important features.
        test_size (int, optional): Size of the test set. If None, no test set is created.
        
    Returns:
        y (pandas.DataFrame): Target data
        features (array-like): Feature indices used
        X_test, y_test (pandas.DataFrame, optional): Test sets
    """
    n, p = X.shape
    snr = 10    
    
    inds = np.random.choice(range(p), s, replace=False)

    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X

    y = snr * np.sum(np.abs(X_np[:, inds]), axis=1) + np.random.normal(size=(n))
    y = pd.DataFrame(y)

    if test_size is not None:
        X_test = np.random.normal(size=(test_size, p))
        y_test = snr * np.sum(np.abs(X_test[:, inds]), axis=1) + np.random.normal(size=(test_size))
        X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
        return y, inds, X_test, y_test
    
    return y, inds

def get_hat_p(y):
    if isinstance(y, torch.Tensor):
        y = y.to(torch.int64)
    else:
        y = torch.tensor(y, dtype=torch.int64) 

    n_items = len(y)
    n_classes = torch.max(y) + 1  # Le nombre de classes est le maximum de y plus 1
    class_counts = torch.zeros(n_classes, dtype=torch.float)

    for class_index in range(n_classes):
        class_counts[class_index] = (y == class_index).sum()

    hat_p = class_counts / n_items
    return hat_p