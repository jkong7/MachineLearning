import numpy as np
# Note: do not import additional libraries to implement these functions


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    confusion_matrix = np.zeros((2,2), dtype=int)
    confusion_matrix[0][0] = np.sum((actual==0) & (predictions==0))
    confusion_matrix[0][1] = np.sum((actual==0) & (predictions==1))
    confusion_matrix[1][0] = np.sum((actual==1) & (predictions==0))
    confusion_matrix[1][1] = np.sum((actual==1) & (predictions==1))

    return confusion_matrix


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    confusion_matrix = compute_confusion_matrix(actual, predictions)
    true_values = confusion_matrix[0][0]+confusion_matrix[1][1]
    total = true_values + confusion_matrix[0][1] + confusion_matrix[1][0]
    return true_values / total 


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """

    confusion_matrix = compute_confusion_matrix(actual, predictions)

    total_positives = confusion_matrix[0][1] + confusion_matrix[1][1]
    true_positives = confusion_matrix[1][1]
    precision = np.nan if total_positives==0 else true_positives / total_positives

    actual_positives = confusion_matrix[1][0] + confusion_matrix[1][1]
    recall = np.nan if actual_positives==0 else true_positives / actual_positives

    return (precision, recall)


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    precision, recall = compute_precision_and_recall(actual, predictions) 
    if precision==np.nan and recall==np.nan: 
        return np.nan 
    if precision==0 and recall==0: 
        return 0 
    return (2*precision*recall) / (precision + recall)
    
    
