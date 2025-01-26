import numpy as np
from src.numpy_practice import find_mode
from src.metrics import compute_accuracy


class Node():
    def __init__(self, return_value=None, split_value=None,
                 attribute_name=None, attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.

        If this is a leaf node, return_value must hold the predicted class.
            In a leaf node, branches is an empty list, and all of
            attribute_name, attribute_index, and split_value should be None.

        If this is not a leaf node, return_value should be None.
            In non-leaf node, branches should be a list of Node objects,
            and all of attribute_name, attribute_index, and split_value
            should have non-None values.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for
                non-leaf nodes) or 0 (at a leaf node).
            attribute_name (str): If not a leaf, contains name of attribute
                that the tree splits the data on. Used for visualization (see
                `DecisionTree.visualize`).
            attribute_index (float): If not a leaf, contains the index of the
                feature vector for the given attribute. Should correspond to
                self.attribute_name.
            split_value (int or float): If not a leaf, contains the value that
                data should be compared to along the given attribute.
            return_value (int): If this is a leaf, the value that this node
                should return.
        """

        if branches is None:
            branches = []
        self.branches = branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.split_value = split_value
        self.return_value = return_value


class DecisionTree():
    def __init__(self, attribute_names):
        """
        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if branch is None:
            branch = self.tree
        self._visualize_helper(branch, level)

        if len(branch.branches) > 0:
            left, right = branch.branches
            if left is not None:
                self.visualize(left, level + 1)

            if left is not None and right is not None:
                tab_level = "  " * level
                print(f"{level}: {tab_level} else:")

            if right is not None:
                self.visualize(right, level + 1)

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        if len(tree.branches) == 0:
            print(f"{level}: {tab_level} Predict {tree.return_value}")
        elif len(tree.branches) == 2:
            print(f"{level}: {tab_level} if {tree.attribute_name} <= {tree.split_value:.1f}:")

    def fit(self, features, labels):
        """
        Takes in the features as a numpy array and fits a decision tree to the labels.
        You shouldn't need to edit this function, but you need to implement the
        self._create_tree function that is called.

        Args:
            features (np.array): numpy array of size NxD containing features, where N is
                number of examples and D is number of features (attributes).
            labels (np.array): numpy array containing class labels for each of the N
                examples.
        Returns:
            None: It should update self.tree with a built decision tree.
        """
        self._check_input(features)

        self.tree = self._create_tree(
            features=features,
            labels=labels,
            used_attributes=[],
            default=0,
        )

    def _create_tree(self, features, labels, used_attributes, default):

        if features.size == 0 or labels.size == 0:
            return Node(return_value=int(default), split_value=None, attribute_name=None, attribute_index=None, branches=[])

        if len(np.unique(labels)) == 1:
            return Node(return_value=int(labels[0].item()), split_value=None, attribute_name=None, attribute_index=None, branches=[])

        if len(used_attributes) == features.shape[1]:
            return Node(return_value=int(find_mode(labels).item()), split_value=None, attribute_name=None, attribute_index=None, branches=[])

        max_ig = -1
        best_attribute = None
        for attribute_index in range(features.shape[1]):
            if attribute_index not in used_attributes:
                ig = information_gain(features, attribute_index, labels)
                if ig > max_ig:
                    max_ig, best_attribute = ig, attribute_index

        if max_ig == -1:
            return Node(return_value=find_mode(labels), split_value=None, attribute_name=None, attribute_index=None, branches=[])

        split_value = 0.5 if len(np.unique(features[:, best_attribute])) == 2 else np.median(features[:, best_attribute])

        left_split_columns = features[:, best_attribute] <= split_value
        right_split_columns = features[:, best_attribute] > split_value

        left_features, left_labels = features[left_split_columns], labels[left_split_columns]
        right_features, right_labels = features[right_split_columns], labels[right_split_columns]

        additional_used_attributes = used_attributes + [best_attribute]

        return Node(
            return_value=None,
            split_value=split_value,
            attribute_name=self.attribute_names[best_attribute],
            attribute_index=best_attribute,
            branches=[self._create_tree(left_features, left_labels, additional_used_attributes, default=find_mode(labels)),
                      self._create_tree(right_features, right_labels, additional_used_attributes, default=find_mode(labels))])
        


    def predict(self, features):
        """
        Predicts label for each example in features using the trained model.

        For example, if you have a decision tree that's visualized as:
            0:  if Outlook <= 0.5:
            1:    if Temp <= 0.5:
            2:      Predict 1
            1:    else:
            2:      Predict 0
            0:  else:
            1:    Predict 1

        then if `tree.attributes = ["Outlook", "Temp"]` and
            `features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`,
            then `tree.predict(features)` should return:
            `array([1, 0, 1, 1])`

        Args:
            features (np.array): numpy array of shape (n, d)
                where n is number of examples and d is number of features (attributes).
        Returns:
            predictions (np.array): numpy array of size N array which has the predicitons
                for the input data.
        """

        predictions = []
        
        for example in features: 
            current=self.tree 
            while len(current.branches) > 0: 
                if example[current.attribute_index] <= current.split_value: 
                    current = current.branches[0]
                else: 
                    current = current.branches[1]
            predictions.append(current.return_value)

        return np.array(predictions).reshape(-1, 1)



def information_gain(features, attribute_index, labels):
    """
    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. For this implementation, we will actually use
    "accuracy gain" which is slightly different than what you may find in
    other writeups about the ID3 algorithm.

    Whenever we split our decision tree, we want to choose that split so
    that the resulting tree has higher accuracy than the tree without that
    split.

    Suppose we have datapoints S and labels L and we are considering splitting
    on attribute A. If A is a binary attribute (it only takes values 0 and 1),
    we will split such that the 0s go to the left and the 1s go to the right.
    If A is continuous, we will split so that A values less than or equal to
    the median go to the left, and A values greater than the median go to the
    right. 

    We'll split the entire dataset S into two subsets:

    For binary A: S(A == 0) and S(A == 1)
    For continuous A: S(A <= m) and S(A > m), where m is the median of A in S.

    Together, the two subsets make up S. If the attribute A were perfectly
    correlated with the class of each data point in S, then all points in a
    given subset will have the same class, and our tree will get 100% accuracy
    on S after just that one split.  Clearly, in this case, we want something
    that captures that A is a good attribute to use in the decision tree. This
    something is information gain. For binary A, this is:

        IG(S, A) = Size(S(A=1)) / Size(S) * Accuracy(S(A=1))
                  + Size(S(A=0)) / Size(S) * Accuracy(S(A=0))
                  - Accuracy(S)

    Where Size(S) is the total number of examples, Size(A=a) is the number of
    examples where attribute A takes value a, and `Accuracy(S)` means, "if we
    had to guess either 1 or 0 for this entirel dataset, what percent would we
    get right?" 

    Hint: you can use `find_mode` and `compute_accuracy` to implement this
    Accuracy function.

    Hint: if Size(S(A=a)) is 0, then just assume Accuracy(S(A=a)) is 1.

    Args:
        features (np.array): numpy array containing features (attributes) for
            each example.
        attribute_index (int): which column of features to use
        labels (np.array): numpy array containing labels corresponding to each example.

    Returns:
        information_gain (float): information (accuracy) gain if the features
            were split on the attribute_index.
    """
    rows = features.shape[0]
    size_s = rows 
    binary = len(np.unique(features[:, attribute_index])) == 2

    if binary: 
        left_split_columns = features[:, attribute_index] == 0
        right_split_columns = features[:, attribute_index] == 1

        left_split_features = features[left_split_columns]
        left_split_labels = labels[left_split_columns]

        right_split_features = features[right_split_columns]
        right_split_labels = labels[right_split_columns]
    else: 
        median_attribute_column = np.median(features[:, attribute_index])
        left_split_columns = features[:, attribute_index] <= median_attribute_column
        right_split_columns = features[:, attribute_index] > median_attribute_column

        left_split_features = features[left_split_columns]
        left_split_labels = labels[left_split_columns]

        right_split_features = features[right_split_columns]
        right_split_labels = labels[right_split_columns]

    size_s_left_split, size_s_right_split = left_split_features.shape[0], right_split_features.shape[0]

    if left_split_labels.size > 0:
        mode_left_split = find_mode(left_split_labels)
        predictions_left_split = np.full_like(left_split_labels, mode_left_split)
        accuracy_left_split = compute_accuracy(left_split_labels, predictions_left_split)
    else:
        accuracy_left_split = 1

    if right_split_labels.size > 0:
        mode_right_split = find_mode(right_split_labels)
        predictions_right_split = np.full_like(right_split_labels, mode_right_split)
        accuracy_right_split = compute_accuracy(right_split_labels, predictions_right_split)
    else:
        accuracy_right_split = 1

    mode_s = find_mode(labels)
    predictions_s = np.full_like(labels, mode_s)
    accuracy_s = compute_accuracy(labels, predictions_s)

    IG = ((size_s_left_split / size_s) * accuracy_left_split) + ((size_s_right_split / size_s) * accuracy_right_split) - accuracy_s

    return IG 


if __name__ == '__main__':
    # Manually construct a simple decision tree and visualize it
    attribute_names = ['Outlook', 'Temp']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    root = Node(
        attribute_name="Outlook", attribute_index=0,
        split_value=0.5, branches=[])

    left = Node(
        attribute_name="Temp", attribute_index=1,
        split_value=0.5, branches=[])

    left_left = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left_right = Node(
        attribute_name=None, attribute_index=None,
        return_value=0, branches=[])

    right = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left.branches = [left_left, left_right]
    root.branches = [left, right]
    decision_tree.tree = root

    decision_tree.visualize()
    # This call should output:
    # 0:  if Outlook <= 0.5:
    # 1:    if Temp <= 0.5:
    # 2:      Predict 1
    # 1:    else:
    # 2:      Predict 0
    # 0:  else:
    # 1:    Predict 1

    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print(tree.predict(features))
    # This should return array([1, 0, 1, 1])
