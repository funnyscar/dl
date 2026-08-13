import numpy as np
import numpy.linalg as LA
import pickle
from PIL import Image

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
    y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
    x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
    y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')
    return x_train, y_train, x_test, y_test

def visualize_data(images: np.ndarray, controls: np.ndarray) -> None:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
    """
    # Current images are in float32 format with values between 0.0 and 255.0
    # Just for the purposes of visualization, convert images to uint8
    images = images.astype(np.uint8)

    indexes = [0, 10, 20]
    for i in indexes:
        im = Image.fromarray(images[i])
        im.save("{}.png".format(i))
        print("vector: ", controls[i])
    


def compute_data_matrix(images: np.ndarray, controls: np.ndarray, standardize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
        standardize (bool): boolean flag that specifies whether the images should be standardized or not

    Returns:
        X (ndarray): input array of size (n, 2700) where each row is the flattened image images[i]
        Y (ndarray): label array of size (n, 3) where row i corresponds to the control for X[i]
    """
    X = images.reshape(len(images), 2700)
    if standardize:
        X = X/255 * 2 - 1
    Y = controls
    return X, Y




def ridge_regression(X: np.ndarray, Y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        lmbda (float): ridge regression regularization term

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """

    I = lmbda * np.identity(2700)
    pi = LA.inv(X.T @ X + I) @ X.T @ Y
    return pi



def ordinary_least_squares(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """ 
    pi = LA.inv(X.T @ X) @ X.T @ Y
    return pi


def measure_error(X: np.ndarray, Y: np.ndarray, pi: np.ndarray) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        pi (ndarray): learned policy of size (2700, 3)

    Returns:
        error (float): the mean Euclidean distance error across all n samples
    """
    return np.average(LA.norm(Y - X @ pi)**2)


def compute_condition_number(X: np.ndarray, lmbda: float) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        lmbda (float): ridge regression regularization term

    Returns:
        kappa (float): condition number of the input array with the given lambda
    """
    M = X.T @ X + lmbda * np.identity(2700)
    U, S, VT = LA.svd(M)
    kappa = np.max(S)/np.min(S)
    return kappa


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()
    print("successfully loaded the training and testing data")

    LAMBDA = [0.1, 1.0, 10.0, 100.0, 1000.0]

    # (a)
    visualize_data(x_train, y_train)

    # (b)
    X, Y = compute_data_matrix(x_train, y_train)
    print(X.shape, Y.shape)

    # pistar = np.matmul(LA.inv(np.matmul(X.T,X)), X.T)
    # print(pistar)

    print("c")
    # (c)
    for l in LAMBDA:
        pi = ridge_regression(X, Y, lmbda=l)
        err = measure_error(X, Y, pi)
        print(err)

    print("d")
    # (d)
    X_norm, Y_norm = compute_data_matrix(x_train, y_train, standardize=True)
    for l in LAMBDA:
        pi = ridge_regression(X_norm, Y_norm, lmbda=l)
        err = measure_error(X_norm, Y_norm, pi)
        print(err)


    print("e")

    X_test, Y_test = compute_data_matrix(x_train, y_train)
    print("without standardization")
    for l in LAMBDA:
        pi = ridge_regression(X_norm, Y_norm, lmbda=l)
        err = measure_error(X_norm, Y_norm, pi)
        print(err)
    print("with standardization")
    X_test_norm, Y_test_norm = compute_data_matrix(x_test, y_test, standardize=True)
    for l in LAMBDA:
        pi = ridge_regression(X_norm, Y_norm, lmbda=l)
        err = measure_error(X_norm, Y_norm, pi)
        print(err)







    