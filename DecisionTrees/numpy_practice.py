import numpy as np


def hello_world():
    """
    In this and future coding assignments, everywhere you see a `raise
    NotImplementedError` you should delete it and write code specified by the
    documentation. There will be at least one test case in the `tests/`
    directory that will check to see whether your implementation provides
    the expected behavior.

    When you run `python -m pytest`, the autograder will print out
    which tests you pass or fail. You should run this often, it will
    simply output `NotImplementedError` for the tests you haven't
    started working on. If you want to check a single test case,
    you can run e.g., `python -m pytest -k test_hello_world`.

    For this test, you should just return the text "Hello, world!"

    Args: None
    Returns: "Hello, world!"
    """

    return "Hello, world!"



def replace_nonfinite_in_place(x):
    """
    In the given array, replace (in-place!) all non-finite values with the value 0.
    You should be able to do this in a single line of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 3, np.inf, 4, 5, np.nan, -1 / 0, 6])
        >>> replace_nans_in_place(x)
        >>> x
        array([1, 2, 3, 0, 4, 5, 0, 0, 6])

    You should use:
        - np.isfinite: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
    You should use one of the following:
        - np.logical_not: https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
        - np.invert: https://numpy.org/doc/stable/reference/generated/numpy.invert.html
        - The tilde (~) syntax, which works as shorthand for np.invert.
    Args:
        x: a numpy array
    Returns:
        None: you can explicitly call `return None`, but Python does so automatically
              if you don't return anything else
    """
    x[np.logical_not(np.isfinite(x))]=0 


def replace_nans_out_of_place(x):
    """
    In the given array, replace (not in-place!) all nans with the value 0.
    You should be able to do this in a single line of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 3, np.inf, 4, 5, np.nan, -1 / 0, 6])
        >>> replace_nans_in_place(x)
        >>> x
        array([1, 2, 3, inf, 4, 5, 0, -inf, 6])
    
    You should use:
        - np.isnan: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        - np.where: https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Args:
        x: a numpy array
    Returns:
        a numpy array where all NaNs (but not infinite) values are replaced with 0s
    """

    return np.where(np.isnan(x), 0, x)


def find_mode(x):
    """
    In the given array, find the mode of the vector.
    The mode is the value that appears the most times (don't worry about ties).

    You should be able to do this in three or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 2, 3, 3, 3])
        >>> find_mode(x)
        3

    You should use:
        - np.argmax: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    You should use one of:
        - np.unique: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        - np.bincount: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

    Args:
        x: a numpy array of integers
    Returns:
        the mode of x
    """
    unique_values, counts_unique_values = np.unique(x, return_counts=True)
    return unique_values[np.argmax(counts_unique_values)]


def flip_and_slice_matrix(x):
    """
    Take the matrix x and flip it horizontally, then take the every third row.

    You should be able to do this in two or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([[ 0,  1,  2],
        ...               [ 3,  4,  5],
        ...               [ 6,  7,  8],
        ...               [ 9, 10, 11],
        ...               [12, 13, 14]])
        >>> flip_and_slice_matrix(x)
        array([[2, 1, 0],
               [11, 10, 9]])

    First, read:
        - https://numpy.org/doc/stable/user/basics.indexing.html#basics-indexing
        
    Args:
      x: a matrix
    Returns:
      a numpy matrix
    """
    return x[:, ::-1][::3, :]


def divide_matrix_along_rows(x, y):
    """
    Take the matrix x and divide it by the vector y, such that
        the ith row of x is divided by the ith value of y.

    You should be able to do this in two or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([[ 0,  1,  2,  3], 
        ...               [ 4,  5,  6,  7],
        ...               [ 8,  9, 10, 11]])
        >>> y = np.array([1, 2, 4])
        >>> divide_rows(x, y)
        array([[0.  , 1.  , 2.  , 3.  ],
               [2.  , 2.5 , 3.  , 3.5 ],
               [2.  , 2.25, 2.5 , 2.75]])

    First, read:
        - https://numpy.org/doc/stable/user/basics.broadcasting.html
    You should use one of:
        - np.reshape: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        - np.newaxis: https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis

    Args:
      x: a matrix
      y: a vector with as many entries as x has rows
    Returns:
      a numpy matrix
    """
    return x/y[:, np.newaxis]
    
