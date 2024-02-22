"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from typing import Type, Dict
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
    ShuffleSplit
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import utils as u 


#1B: checking for X
def scalex(x_check = NDArray[np.floating]):
    # checking if all elements are floats and within the range of [0, 1]: 
    return issubclass(x_check.dtype.type, np.floating) and not ((x_check < 0).any() or (x_check > 1).any())

#1B: checking for Y
def scaley(y_check: NDArray[np.integer]):
   # checking if all the elements in y are integers
    return issubclass(y_check.dtype.type, np.integer)
    
    
def print_cv_result_dict_1(cv_dict):
    # filtering out the keys where we don't calculate mean and std
    filtered_keys = [k for k in cv_dict if k not in ['fit_time', 'score_time']]
    
    # Iterate over the filtered keys and print the mean and std for the arrays
    for key in filtered_keys:
        mean_value = cv_dict[key].mean()
        std_value = cv_dict[key].std()
        print(f"mean_{key}: {mean_value}, std_{key}: {std_value}")

def extract_scores(cv_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {
            'mean_fit_time': cv_results['fit_time'].mean(),
            'std_fit_time': cv_results['fit_time'].std(),
            'mean_accuracy': cv_results['test_score'].mean(),
            'std_accuracy': cv_results['test_score'].std()
        }


def filter_imbal_7_9s(X, y):
    
    indices_of_7_and_9 = (y == 7) | (y == 9)
    X_filtered = X[indices_of_7_and_9]
    y_filtered = y[indices_of_7_and_9]
    
    #cnverting 7 to 0 and 9 to 1
    y_binary = np.where(y_filtered == 7, 0, 1)
    
    # Identify indices for class 9 (now labeled as 1)
    indices_of_class_9 = np.where(y_binary == 1)[0]
    
    # number of class 9 instances
    number_to_remove = int(len(indices_of_class_9) * 0.9)
    
    np.random.shuffle(indices_of_class_9)
    
    #indices to remove
    indices_to_remove = indices_of_class_9[:number_to_remove]
    
    # indices to keep
    indices_to_keep = np.setdiff1d(np.arange(len(X_filtered)), indices_to_remove)
    
    # imbalanced data
    X_imbal = X_filtered[indices_to_keep]
    y_imbal = y_binary[indices_to_keep]
    
    return X_imbal, y_imbal
