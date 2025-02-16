import numpy as np


def mean_squared_error(predictions, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated.

    Refer to the slides, or read more here:
      https://en.wikipedia.org/wiki/Mean_squared_error

    Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

    Args:
        predictions (np.ndarray): the predicted values
        targets (np.ndarray): the ground truth values

    Returns:
        MSE (float): the mean squared error across all predictions and targets
    """
    assert predictions.shape == targets.shape
    if len(predictions.shape)==2: 
        rows, cols = predictions.shape
    else: 
        rows, cols = predictions.shape[0], 1
    total = rows * cols 
    mse_sum = np.sum((predictions-targets)**2)
    return mse_sum / total 



def demographic_parity_difference(confusion_matrix_a, confusion_matrix_b):
    """
    A classifier satisfies demographic parity if the subjects in the protected
    and unprotected groups have equal probability of being assigned to the
    positive predicted class.
    https://en.wikipedia.org/wiki/Fairness_(machine_learning)#Definitions_based_on_predicted_outcome

    You can assume each confusion_matrix input is of shape (2, 2), where the
    entries are:
    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    >>> demographic_parity_difference(
    ...     np.array([[10, 0], [0, 10]]), np.array([[10, 0], [0, 10]]))
    0.0

    >>> demographic_parity_difference(
    ...     np.array([[9, 13], [10, 18]]), np.array([[15, 14], [9, 12]]))
    0.1

    Args:
        confusion_matrix_a (np.ndarray): a 2x2 confusion matrix, for members of "group a"
        confusion_matrix_b (np.ndarray): a 2x2 confusion matrix, for members of "group b"

    Returns:
        DPD: The demographic parity difference between group a and group b

    """
    mat_a_shape = confusion_matrix_a.shape
    assert mat_a_shape == confusion_matrix_b.shape
    assert mat_a_shape == (2, 2)

    total_a, total_b = np.sum(confusion_matrix_a), np.sum(confusion_matrix_b)

    positive_classification_a, positive_classification_b = confusion_matrix_a[0][1] + confusion_matrix_a[1][1], confusion_matrix_b[0][1] + confusion_matrix_b[1][1]
    
    dem_parity_a, dem_parity_b = positive_classification_a / total_a, positive_classification_b / total_b

    return abs(dem_parity_a - dem_parity_b)


def equalized_odds_difference(confusion_matrix_a, confusion_matrix_b):
    """
    A classifier satisfies equalized odds if the subjects in the protected and
    unprotected groups have equal TPR and equal FPR
    https://en.wikipedia.org/wiki/Equalized_odds

    We define equalized odds difference (EOD) as the maximum absolute disparity between
    either false-positive rates or true-positive rates between groups.

    You can assume each confusion_matrix input is of shape (2, 2), where the
    entries are:
    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    >>> equalized_odds_difference(
    ...     np.array([[10, 0], [0, 10]]), np.array([[10, 0], [0, 10]]))
    0.0

    >>> equalized_odds_difference(
    ...     np.array([[9, 13], [10, 18]]), np.array([[15, 14], [11, 10]]))
    0.1666666666666667

    Args:
        confusion_matrix_a (np.ndarray): a 2x2 confusion matrix, for members of "group a"
        confusion_matrix_b (np.ndarray): a 2x2 confusion matrix, for members of "group b"

    Returns:
        EOD: The equalized odds difference between group a and group b

    """
    mat_a_shape = confusion_matrix_a.shape
    assert mat_a_shape == confusion_matrix_b.shape
    assert mat_a_shape == (2, 2)

    FPR_a, FPR_b = confusion_matrix_a[0][1] / (confusion_matrix_a[0][1] + confusion_matrix_a[0][0]), confusion_matrix_b[0][1] / (confusion_matrix_b[0][1] + confusion_matrix_b[0][0])
    FPRD = abs(FPR_a - FPR_b)

    TPR_a, TPR_b = confusion_matrix_a[1][1] / (confusion_matrix_a[1][0] + confusion_matrix_a[1][1]), confusion_matrix_b[1][1] / (confusion_matrix_b[1][0] + confusion_matrix_b[1][1])
    TPRD = abs(TPR_a - TPR_b)
    return max(FPRD, TPRD)
