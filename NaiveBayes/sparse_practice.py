import numpy as np


def sparse_to_numpy(X):
    """
    Convert a sparse matrix to a numpy array.

    X is a *sparse* matrix, which makes it more efficient than a typical numpy
        array when most elements are 0. If you call `print(X)`, it will display
        its contents in the form `(a, b) c` which means that `X[a, b] == c`.

    For example:
        >>> arr = np.arange(100).reshape(5, 20)
        >>> mat = csr_matrix(arr)
        >>> arr2 = sparse_to_numpy(mat)
        >>> assert np.allclose(arr, arr2)

    First, read the documentation on sparse matrices:
        https://docs.scipy.org/doc/scipy/reference/sparse.html

    Then, write a single line of code that converts X into a numpy array. You
        may use functions that are described in the documentation above.
        Do not use a for or while loop.
        While you can and should use methods that are already defined for X,
        Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

    Args:
        X: 2D sparse matrix
    Returns:
        2D numpy matrix with the same data as X
    """
    return X.toarray()



def sparse_multiplication(X, Y):
    """
    Perform matrix multiplication with X and Y.

    X is a *sparse* matrix, which makes it more efficient than a typical numpy
        array when most elements are 0. If you call `print(X)`, it will display
        its contents in the form `(a, b) c` which means that `X[a, b] == c`.

    Y may be either another sparse matrix or a standard numpy vector.

    For example:
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A
        <3x3 sparse matrix of type '<class 'numpy.int64'>'
              with 5 stored elements in Compressed Sparse Row format>
        >>> print(A)
          (0, 0)	1
          (0, 1)	2
          (1, 2)	3
          (2, 0)	4
          (2, 2)	5
        >>> v = np.array([1, 0, -1])
        >>> sparse_multiplication(A, v)
        array([ 1, -3, -1], dtype=int64)

    First, read the documentation on sparse matrices:
        https://docs.scipy.org/doc/scipy/reference/sparse.html

    Then, write a single line of code that multiplies X and Y. You
        may use functions that are described in the documentation above.
        Do not use a for or while loop.
        While you can and should use methods that are already defined for X,
        Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

    Args:
        X: 2D sparse matrix
        Y: 2D sparse matrix or 2D numpy array
    Returns:
        2D sparse matrix or 2D numpy array resulting from multiplying X and Y
    """
    return X @ Y



def flip_bits_sparse_matrix(X):
    """
    This is a helper function you do not need to modify.
    You may find it helpful in certain Naive Bayes functions.
    Given a matrix of ones and zeros, this flips all zeros to ones and vice versa.

    For example:
        >>> D = np.array([1, 0, 0, 1])
        >>> X = csr_matrix(D)
        >>> print(X)
          (0, 0)	1
          (0, 3)	1
        >>> Y = flip_bits_sparse_matrix(X)
        >>> print(Y)
          (0, 1)	1
          (0, 2)	1

    Args:
        X: 2D sparse binary matrix
    Returns:
        2D sparse binary matrix representing `1 - X`
    """
 
    X_type = str(type(X))[21:36]
    msg = f"X should be sparse binary matrix"
    assert X_type == "_csr.csr_matrix", msg
    assert np.all((X.data == 0) | (X.data == 1)), msg
    one_minus_X = np.array(np.ones(X.shape) - X.todense(), dtype=X.dtype)

    return X.__class__(one_minus_X, shape=X.shape, dtype=X.dtype)