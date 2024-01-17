import numpy as np
from sklearn.metrics import mean_squared_error


def normalized_rmse(y_true, y_pred):
    return 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)
