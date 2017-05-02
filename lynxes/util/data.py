import numpy as np
#import pandas as pd


def add_column_numpy_array(array, new_col):
    placeholder = np.ones(array.shape[0])[:, np.newaxis]
    result = np.hstack((array, placeholder))

    if isinstance(new_col, np.ndarray):
        assert array.shape[0] == new_col.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(array.shape[0],
                                                                        new_col.shape[0])
        assert len(new_col.shape) <= 2, "new column must be 1D or 2D"

        if len(new_col.shape) == 1:
            new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    elif isinstance(new_col, list):
        assert len(new_col) == array.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(len(array),
                                                                        len(new_col))
        new_col = np.array(new_col)
        assert len(new_col.shape) == 1, "list elements cannot be iterable"
        new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    else:
        placeholder = np.ones(array.shape[0])[:, np.newaxis]
        result = np.hstack((array, placeholder))
        result[:, -1] = new_col
        return result
