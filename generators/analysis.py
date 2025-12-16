import numpy as np
from file import get_predictions

print(get_predictions("crtxn1", "g25f0"))

def hamming(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape.")
    return np.sum(arr1 != arr2)