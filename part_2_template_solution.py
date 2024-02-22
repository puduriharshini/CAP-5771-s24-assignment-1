# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_score,
    KFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}

        # Enter your code and fill the `answer`` dictionary
        print("Part 2A")
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        test_Xtrain = nu.scalex(Xtrain)
        test_Xtest = nu.scalex(Xtest)
        test_ytrain = nu.scaley(ytrain)
        test_ytest = nu.scaley(ytest)
        print("Are the Xtrain elements scaled from 0 to 1 and represented as floating point numbers?  " +str(test_Xtrain))
        print("Are the Xtest elements scaled from 0 to 1 and represented as floating point numbers?  " +str(test_Xtest))
        print("Are the ytrain elements represented as integers? " +str(test_ytrain))
        print("Are the ytest elements represented as integers?" +str(test_ytest))
        
        #number of elements in each class
        uniq_cls_train, class_count_train = np.unique(ytrain, return_counts=True)
        uniq_cls_test, class_count_test = np.unique(ytest, return_counts=True)
        

        # Construct the answer dictionary with the required keys
        answer = {
            "nb_classes_train": len(uniq_cls_train),
            "nb_classes_test": len(uniq_cls_test),
            "class_count_train": class_count_train,
            "class_count_test": class_count_test,
            "length_Xtrain": Xtrain.shape[0],
            "length_Xtest": Xtest.shape[0],
            "length_ytrain": ytrain.shape[0],
            "length_ytest": ytest.shape[0],
            "max_Xtrain": Xtrain.max(),
            "max_Xtest": Xtest.max()
        }
        # debugging and verification
        print(f"Number of classes in training set: {answer['nb_classes_train']}")
        print(f"Class distribution in training set: {answer['class_count_train']}")
        print(f"Number of classes in testing set: {answer['nb_classes_test']}")
        print(f"Class distribution in testing set: {answer['class_count_test']}")
        print(f"Number of samples in Xtrain: {answer['length_Xtrain']}")
        print(f"Number of samples in Xtest: {answer['length_Xtest']}")
        print(f"Number of labels in ytrain: {answer['length_ytrain']}")
        print(f"Number of labels in ytest: {answer['length_ytest']}")
        print(f"Maximum value in Xtrain: {answer['max_Xtrain']}")
        print(f"Maximum value in Xtest: {answer['max_Xtest']}")

        return answer, Xtrain, ytrain, Xtest, ytest
    
    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}
        print("2B")

        X, y, Xtest, ytest = u.prepare_data()
        
        answer = {}
        

        for ntrain, ntest in zip(ntrain_list, ntest_list):
            # Split the data
            X_train = X[0:ntrain, :]
            y_train = y[0:ntrain]
            X_test = Xtest[0:ntest, :]
            y_test = ytest[0:ntest]

            # part C: cv with logistic regression
            clf_C = DecisionTreeClassifier(random_state=52)
            cv_C = KFold(n_splits=5, shuffle = True, random_state=52)
            scores_C = u.train_simple_classifier_with_cv(Xtrain = X_train, ytrain = y_train, clf = clf_C, cv=cv_C)
            mean_fit_time_C = np.mean(scores_C['fit_time'])
            std_fit_time_C = np.std(scores_C['fit_time'])
            mean_acc_C = np.mean(scores_C['test_score'])
            std_acc_C = np.std(scores_C['test_score'])

            # part D: train and testing the split evaluation
            clf_D = DecisionTreeClassifier(random_state=52)
            cv_D = ShuffleSplit(n_splits=5, random_state=52)
            scores_D = u.train_simple_classifier_with_cv(Xtrain = X_train, ytrain = y_train, clf = clf_D, cv=cv_D)
            mean_fit_time_D = np.mean(scores_D['fit_time'])
            std_fit_time_D = np.std(scores_D['fit_time'])
            mean_acc_D = np.mean(scores_D['test_score'])
            std_acc_D = np.std(scores_D['test_score'])

            # part F: 
            clf_F = LogisticRegression(max_iter=300)
            cv_F = ShuffleSplit(n_splits=5, random_state=52)
            #training
            clf_F.fit(X_train, y_train)
            #predictions
            y_pred_train_F = clf_F.predict(X_train)
            y_pred_test_F = clf_F.predict(X_test)
            #calculating scores
            scores_train_F = accuracy_score(y_train, y_pred_train_F)
            scores_test_F = accuracy_score(y_test, y_pred_test_F)
            # calculating mean cross-validation accuracy
            cross_val_scores_F = cross_val_score(clf_F, X_train, y_train, cv=cv_F)
            mean_cv_accuracy_F = cross_val_scores_F.mean()
            #confusion matrices for train and test data
            conf_mat_train = confusion_matrix(y_train, y_pred_train_F)
            conf_mat_test = confusion_matrix(y_test, y_pred_test_F)

            # compiling results
            partC = {"clf": clf_C, "cv": cv_C, "scores": {"mean_fit_time": mean_fit_time_C, "std_fit_time": std_fit_time_C, "mean_accuracy": mean_acc_C, "std_accuracy": std_acc_C}}
            partD = {"clf": clf_D, "cv": cv_D, "scores": {"mean_fit_time": mean_fit_time_D, "std_fit_time": std_fit_time_D, "mean_accuracy": mean_acc_D, "std_accuracy": std_acc_D}}
            partF = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": clf_F,
                "cv": cv_F,
                "conf_mat_train": conf_mat_train.tolist(),
                "conf_mat_test": conf_mat_test.tolist(),
            }

            answer[ntrain] = {
                "partC": partC,
                "partD": partD,
                "partF": partF,
                "ntrain": ntrain,
                "ntest": ntest,
                "class_count_train": list(np.bincount(y_train)),
                "class_count_test": list(np.bincount(y_test)),
            }

        print(answer)
        return answer

        
        
        
        
