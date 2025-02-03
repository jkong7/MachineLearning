import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    M, _ = X.shape
    N, _ = Y.shape 
    D=np.zeros((M,N))

    for i in range(M): 
        for j in range(N): 
            D[i][j]=np.linalg.norm(Y[j]-X[i])
    return D
        


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    M, _ = X.shape
    N, _ = Y.shape 
    D=np.zeros((M,N))

    for i in range(M): 
        for j in range(N): 
            D[i][j]=np.linalg.norm(Y[j]-X[i], ord=1)
    return D


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)
    (Hint: this is cosine distance, not cosine similarity)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    M, _ = X.shape
    N, _ = Y.shape 
    D=np.zeros((M,N))

    for i in range(M): 
        for j in range(N): 
            D[i][j]=1 - (np.dot(X[i], Y[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(Y[j])+1e-8))
    return D
