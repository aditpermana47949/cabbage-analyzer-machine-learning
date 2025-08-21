import pickle
import numpy as np


def get_info(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return np.round(model.coef_, 4), model.intercept_
