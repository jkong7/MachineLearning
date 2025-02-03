import numpy as np
import src.random


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.
        Do not use (the name of) these numpy or internal functions: lstsq, polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree

        self.weights = np.ones((self.degree + 1,))

    def fit(self, features, targets):
        """
        Fit the model to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.
          - If you are getting `numpy.linalg.LinAlgError: Singular matrix`,
            you may want to compute a "pseudoinverse" or add a tiny bit of
            random noise to your input data.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model weights to `self.weights`)
        """
        N = features.shape[0]
        features_degree = np.ones((N, self.degree+1))
        for degree in range(1, self.degree+1): 
            features_degree[:, degree] = features[:, 0] ** degree
        self.weights = np.linalg.pinv(features_degree.T @ features_degree) @ (features_degree.T @ targets)

    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this only after calling fit so that the model has its weights.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        assert hasattr(self, "weights"), "Model hasn't been fit!"

        N= features.shape[0]
        features_degree = np.ones((N, self.degree+1))
        for degree in range(1, self.degree+1): 
            features_degree[:, degree] = features[:,0] ** degree 
        predictions = features_degree @ self.weights 
        return predictions.reshape(-1,1)
