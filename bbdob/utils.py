import numpy as np


def idx2one_hot(x, max_size):
    """
    Values, or indexes, is converted to one-hot representations.

    Parameters
    ----------
    x : numpy.ndarray
        A vector or a population of vectors.
    max_size : int
        Maximum cardinality of the input.

    Returns
    -------
    numpy.ndarray
        One-hot representations of x.

    Examples
    --------
    >>> a = np.array([1, 2, 0, 2])
    >>> idx2one_hot(a, 3)
    array([[0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1]])
    >>> b = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    >>> idx2one_hot(b, 2)
    array([[[1, 0],
            [0, 1],
            [0, 1],
            [1, 0]],

           [[0, 1],
            [1, 0],
            [0, 1],
            [0, 1]]])
    """
    assert max_size > 0, \
        "max_size ({}) must be non-negative integer value".format(max_size)
    return np.identity(max_size, dtype=np.int)[x]


def packbits(bin_array, reverse=True):
    """
    Convert binary numbers to decimal numbers.

    Parameters
    ----------
    bin_array : numpy.ndarray
        Bit strings.
    reverse : bool, default True
        If the last element of the list is to be treated as the first bit of the bit string, then True, else False.

    Returns
    -------
    numpy.ndarray
        Decimal numbers converted from binary numbers.

    Examples
    --------
    >>> a = np.array([0, 0, 1, 1])
    >>> packbits(a)
    3
    >>> packbits(a, reverse=False)
    12
    >>> b = np.array([[0, 1, 1, 0], [1, 1, 1, 1]])
    >>> packbits(b)
    array([ 6, 15])
    """
    p = np.power(2, np.arange(bin_array.shape[-1]))
    if reverse:
        p = p[::-1]
    dec_array = np.dot(bin_array, p)
    return dec_array
