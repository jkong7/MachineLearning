import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.sparse_practice import flip_bits_sparse_matrix
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm

    Note that the class definition above indicates that this class
        inherits from the NaiveBayes class. This means it has the same
        functions as the NaiveBayes class unless they are re-defined in this
        function. In particular you should be able to call `self.predict_proba`
        using your implementation from `src/naive_bayes.py`.
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm
            smoothing: controls the smoothing behavior when computing beta
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `p(y_i = k) = 1 / n_labels`
            for all k
        and initialize self.beta such that
            `p(w_j | y_i = k) = 1/2`
            for all j, k.
        """
        self.alpha = np.zeros(n_labels) 
        for k in range(n_labels):
            self.alpha[k] = 1 / n_labels  

        self.beta = np.zeros((vocab_size, n_labels)) 
        for j in range(vocab_size):  
            for k in range(n_labels):  
                self.beta[j, k] = 0.5  

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the NaiveBayes superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* overwrite the provided `y` array with the
            true labels with your predicted labels. 

        During the M-step, update self.alpha and self.beta, similar to the
            `fit()` call from the NaiveBayes superclass. Unlike NaiveBayes,
            you will use unlabeled data. When counting the words in an
            unlabeled document in the computation for self.beta, to replace
            the missing binary label y, you should use the predicted probability
            p(y | X) inferred during the E-step above.

        For help understanding the EM algorithm, refer to the lectures and
            the handout.

        self.alpha should contain the marginal probability of each class label.

        self.beta is an array of shape [n_vocab, n_labels]. self.beta[j, k]
            is the probability of seeing the word j in a document with label k.
            Remember to use self.smoothing. If there are M documents with label
            k, and the `j`th word shows up in L of them, then `self.beta[j, k]`.

        Note: if self.max_iter is 0, your function should call
            `self.initialize_params` and then break. In each
            iteration, you should complete both an E-step and
            an M-step.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size
        self.initialize_params(vocab_size, n_labels)
        if self.max_iter == 0: 
            return 
        
        X_label, X_unlabel, y_label = X[~np.isnan(y)], X[np.isnan(y)], y[~np.isnan(y)].astype(int)

        for i in range(self.max_iter): 
            predictions = self.predict_proba(X_unlabel)
            priors = np.zeros(n_labels)
            for k in range(n_labels):
                labeled = np.sum(y_label == k)
                unlabeled = np.sum(predictions[:, k])
                priors[k] = labeled + unlabeled

            self.alpha = priors / np.sum(priors)

            

            self.beta = np.zeros((vocab_size, n_labels))

            for label in range(n_labels):
                X_labeled_current = X_label[y_label == label]
                num_labeled_docs = X_labeled_current.shape[0]
                word_in_labeled_docs = (X_labeled_current > 0).sum(axis=0)
                weights_for_unlabeled_docs = predictions[:, label][:, np.newaxis]  
                weighted_word_in_unlabeled_docs = X_unlabel.multiply(weights_for_unlabeled_docs)
                word_in_unlabeled_docs_sum = weighted_word_in_unlabeled_docs.sum(axis=0)

                total_occurences = np.asarray(word_in_labeled_docs + word_in_unlabeled_docs_sum).squeeze()

                total_docs = num_labeled_docs + np.sum(predictions[:, label])

                smoothed_num = total_occurences + self.smoothing
                smoothed_denom = total_docs + (2 * self.smoothing)

                self.beta[:, label] = smoothed_num / smoothed_denom

    def likelihood(self, X, y):
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data. You should use
            logs to avoid underflow.  This function *should* use unlabeled
            data.

        For unlabeled data, we predict `p(y_i = y' | X_i)` using the
            previously-learned p(x|y, beta) and p(y | alpha).
            For labeled data, we define `p(y_i = y' | X_i)` as
            1 if `y_i = y'` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.

        The tricky aspect of this likelihood is that we are simultaneously
            computing $p(y_i = y' | X_i, \alpha^t, \beta^t)$ to predict a
            distribution over our latent variables (the unobserved $y_i$) while
            at the same time computing the probability of seeing such $y_i$
            using $p(y_i =y' | \alpha^t)$.

        Note: In implementing this equation, it will help to use your
            implementation of `stable_log_sum` to avoid underflow. See the
            documentation of that function for more details.

        We will provide a detailed writeup for this likelihood in the PDF
            handout.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        X_label, X_unlabel, y_label = X[~np.isnan(y)], X[np.isnan(y)], y[~np.isnan(y)].astype(int)

        priors = np.log(self.alpha[y_label])
        presences = X_label.multiply(np.log(self.beta[:, y_label].T))
        absences = flip_bits_sparse_matrix(X_label).multiply(np.log(1.0 - self.beta[:, y_label].T))
        futures = np.sum(priors + np.sum(presences + absences, axis=1).A1)

        priors_unlabel = np.log(self.alpha)
        presences_unlabel = X_unlabel @ np.log(self.beta)
        absences_unlabel = flip_bits_sparse_matrix(X_unlabel) @ np.log(1.0 - self.beta)
        futures_unlabel = priors_unlabel + presences_unlabel + absences_unlabel
        futures_unlabel_safe = np.sum(stable_log_sum(futures_unlabel))

        likelihoods = futures + futures_unlabel_safe
        return likelihoods