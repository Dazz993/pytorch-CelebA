import numpy as np
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import average_precision_score

np.seterr(divide='ignore', invalid='ignore')

def f1_score(y_pred, y_true):
    '''
    Compute micro f1 score and macro f1 score for multi-label classification outputs

    Args:
        y_pred: the output should have gone through a sigmoid layer
        y_true: true binary labels or binary label indicators

    Returns:

    '''
    assert isinstance(y_pred, np.ndarray), isinstance(y_true, np.ndarray)
    y_pred = (y_pred > 0.5).astype(float)
    f1_micro = sk_f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = sk_f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f1_micro, f1_macro


def map_score(y_pred, y_true):
    '''
    Compute mean average precision

    Args:
        y_pred: the output should have gone through a sigmoid layer
        y_true: true binary labels or binary label indicators

    Returns:

    '''
    # assert isinstance(y_pred, np.ndarray), isinstance(y_true, np.ndarray)
    # map = average_precision_score(y_true, y_pred)
    # return map
    return 0