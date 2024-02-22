# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    KFold
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
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

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        print("1A: "+ str(answer)) #output returns 0
        return answer
    

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        print("1B")
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        X_train, y_train = u.filter_out_7_9s(Xtrain, ytrain)
        X_test, y_test = u.filter_out_7_9s(Xtest, ytest)
        test_Xtrain = nu.scalex(Xtrain)
        test_Xtest = nu.scalex(Xtest)
        
        to ensure that our labels are integers, we test them
        test_ytrain = nu.scaley(ytrain)
        test_ytest = nu.scaley(ytest)
        print("checking if Xtrain elements scaled from 0 to 1 and represented as floating point numbers? " +str(test_Xtrain))
        print("checking if Xtest elements scaled from 0 to 1 and represented as floating point numbers? " +str(test_Xtest))
        print("checking if ytrain elements represented as integers? " +str(test_ytrain))
        print("checking if ytest elements represented as integers? " +str(test_ytest))
        answer = {}

        # Enter your code and fill the `answer` dictionary

        len_Xtrain = len(Xtrain)
        len_Xtest = len(Xtest)
        len_Ytrain = len(ytrain)
        len_Ytest = len(ytest)
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()
        print(f"1B - The lengths of Xtrain, Xtest, ytrain, ytest are: {len_Xtrain}, {len_Xtest}, {len_Ytrain}, {len_Ytest}")
        print(f"1B - The maximum value of Xtrain and Xtest is: {max_Xtrain}, {max_Xtest}")
        answer["len_Xtrain"] = len_Xtrain
        answer["len_Xtest"] = len_Xtest
        answer["len_Ytrain"] = len_Ytrain
        answer["len_Ytest"] = len_Ytest
        answer["max_Xtrain"] = max_Xtrain
        answer["max_Xtest"] = max_Xtest

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        print("Part 1C")
        Xtrain, ytrain, Xtest, ytest = u.prepare_data() # preparing initial dataset
        X_train, y_train = u.filter_out_7_9s(Xtrain, ytrain)
        X_test, y_test = u.filter_out_7_9s(Xtest, ytest)
        clf = DecisionTreeClassifier(random_state=52) #defining clf and cv
        cv = KFold(n_splits=5, shuffle = True, random_state = 52)
        acc_scores1 = u.train_simple_classifier_with_cv(Xtrain=X_train, ytrain=y_train, clf=clf, cv=cv)
        u.print_cv_result_dict(acc_scores1) 
        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state = 52)  
        answer["cv"] = KFold(n_splits=5, shuffle=True, random_state = 52)  
        score_valC={}
        metrics_score = ['fit_time', 'test_score']
        for metric in metrics_score:
            
            if metric in acc_scores1:
                # computing mean and standard deviation for the metric
                mean_value = np.mean(acc_scores1[metric])
                std_value = np.std(acc_scores1[metric])
                
                # updating score_valD dictionary with the computed values
                score_valC[f'mean_{metric}'] = mean_value
                score_valC[f'std_{metric}'] = std_value

        answer["scores"] = score_valC
        return answer



    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        print("Part 1D")
        # Enter your code and fill the `answer` dictionary
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        clf = DecisionTreeClassifier(random_state=52)
        cv = ShuffleSplit(n_splits=5, random_state = 52)
        acc_scores2 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf, cv=cv)
        u.print_cv_result_dict(acc_scores2)
        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state = 52)  
        answer["cv"] = ShuffleSplit(n_splits=5, random_state = 52) 
        score_valD = {}
        metrics_score = ['fit_time', 'test_score']
        for metric in metrics_score:
            
            if metric in acc_scores2:
                # computing mean and standard deviation for the metric
                mean_value = np.mean(acc_scores2[metric])
                std_value = np.std(acc_scores2[metric])
                
                # updating score_valD dictionary with the computed values
                score_valD[f'mean_{metric}'] = mean_value
                score_valD[f'std_{metric}'] = std_value

        answer["scores"] = score_valD
        print("K-Fold splits the data into k fixed subsets and rotates them for training and testing, while ShuffleSplit randomly shuffles and splits the data into training and testing subsets for each iteration.")

        return answer


        


    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`
        print("Part 1E")
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        X_train, y_train = u.filter_out_7_9s(Xtrain, ytrain)
        X_test, y_test = u.filter_out_7_9s(Xtest, ytest)
        
        answer = {}
        # Enter your code, construct the `answer` dictionary, and return it.
        for k in [2, 5, 8, 16]:
            print(f"K={k}: ")
            
            cv = ShuffleSplit(n_splits=k, random_state=52)
            clf = DecisionTreeClassifier(random_state=52)
            
            acc_scores3 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf, cv=cv)
            
            nu.print_cv_result_dict_1(acc_scores3)

            mean_accuracy = acc_scores3['test_score'].mean()
            std_accuracy = acc_scores3['test_score'].std()

            answer[k] = {
                'scores': {
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy
                },
                'cv': cv,
                'clf': clf
            }

        print("Noticing Difference: While the standard deviation does tend to vary, the mean generally stays the same")
        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.
        print("Part 1F")
        X, y, X_test, y_test = u.prepare_data()
        X_train, y_train = u.filter_out_7_9s(X, y)
        X_test, y_test = u.filter_out_7_9s(X_test, y_test)
        
        cv = ShuffleSplit(n_splits=5, random_state=52) 
        # training random forest
        rf_clf = RandomForestClassifier(random_state=52)
        rf_scores = u.train_simple_classifier_with_cv(Xtrain=X_train, ytrain=y_train, clf=rf_clf, cv=cv)


        # training decision trees
        dt_clf = DecisionTreeClassifier(random_state=52)
        dt_scores = u.train_simple_classifier_with_cv(Xtrain=X_train, ytrain=y_train, clf=dt_clf, cv=cv)

        #answer dict
        answer = {
            "clf_RF": rf_clf,
            "clf_DT": dt_clf,
            "cv": cv,
            "scores_RF": nu.extract_scores(rf_scores),
            "scores_DT": nu.extract_scores(dt_scores),
            "model_highest_accuracy": "Random Forest" if rf_scores['test_score'].mean() > dt_scores['test_score'].mean() else "Decision Trees",
            "model_lowest_variance": "Random Forest" if rf_scores['test_score'].std() < dt_scores['test_score'].std() else "Decision Trees",
            "model_fastest": "Random Forest" if rf_scores['fit_time'].mean() < dt_scores['fit_time'].mean() else "Decision Trees"
        }


        print(answer)
        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        print("Part 1G")

        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        base_clf = RandomForestClassifier(random_state=52)
        default_parameters = base_clf.get_params()

        grid_par = {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf' : [1, 2, 3]
            }

        #cross-validation strategy
        shuf_split_cv = ShuffleSplit(n_splits=5, random_state=52) 
        
        
        #grid search with cv
        grid_search = GridSearchCV(RandomForestClassifier(random_state=52), grid_par, cv=shuf_split_cv, scoring='accuracy')
        grid_search.fit(Xtrain, ytrain)


        #extract the info from grid search
        mean_acc_cv = grid_search.best_score_
        print("Mean Accuracy Score from Grid Search Cross-Validation: ", mean_acc_cv)
        best_parameters = grid_search.best_params_
        print("Best Parameters: ", best_parameters)
        best_estimator = grid_search.best_estimator_

        #evaluating the best estimator on training data and test data
        train_pred_orig = base_clf.fit(Xtrain, ytrain).predict(Xtrain)
        test_pred_orig = base_clf.predict(Xtest)
        train_pred_best = best_estimator.predict(Xtrain)
        test_pred_best = best_estimator.predict(Xtest)

        # confusion matrices
        conf_matrix_train_orig = confusion_matrix(ytrain, train_pred_orig)
        conf_matrix_test_orig = confusion_matrix(ytest, test_pred_orig)
        conf_matrix_train_best = confusion_matrix(ytrain, train_pred_best)
        conf_matrix_test_best = confusion_matrix(ytest, test_pred_best)

        # accuracies
        accuracy_train_orig = np.diag(conf_matrix_train_orig).sum() / conf_matrix_train_orig.sum()
        accuracy_test_orig = np.diag(conf_matrix_test_orig).sum() / conf_matrix_test_orig.sum()
        accuracy_train_best = np.diag(conf_matrix_train_best).sum() / conf_matrix_train_best.sum()
        accuracy_test_best = np.diag(conf_matrix_test_best).sum() / conf_matrix_test_best.sum()

        # train the best estimator on the full training set
        best_estimator.fit(Xtrain, ytrain)
        train_accuracy = best_estimator.score(Xtrain, ytrain)
        test_accuracy = best_estimator.score(Xtest, ytest)

        training and testing accuracy
        print(f"The training accuracy of the best estimator: {train_accuracy}")
        print(f"Testing accuracy of the best estimator: {test_accuracy}")

        comments on results
        if train_accuracy > mean_acc_cv:
            print("Training accuracy is higher than the mean CV accuracy.")
        elif train_accuracy < mean_acc_cv:
            print("Training accuracy is lower than the mean CV accuracy.")
        else:
            print("Training accuracy is about the same as the mean CV accuracy.")

        if test_accuracy > mean_acc_cv:
            print("Testing accuracy is higher than the mean CV accuracy.")
        elif test_accuracy < mean_acc_cv:
            print("Testing accuracy is lower than the mean CV accuracy.")
        else:
            print("Testing accuracy is about the same as the mean CV accuracy.")



        # answer dictionary
        answer = {
            "clf": base_clf,
            "default_parameters": default_parameters,
            "best_estimator": best_estimator,
            "grid_search": grid_search,
            "mean_accuracy_cv": mean_acc_cv,
            "confusion_matrix_train_orig": conf_matrix_train_orig,
            "confusion_matrix_train_best": conf_matrix_train_best,
            "confusion_matrix_test_orig": conf_matrix_test_orig,
            "confusion_matrix_test_best": conf_matrix_test_best,
            "accuracy_orig_full_training": accuracy_train_orig,
            "accuracy_best_full_training": accuracy_train_best,
            "accuracy_orig_full_testing": accuracy_test_orig,
            "accuracy_best_full_testing": accuracy_test_best
        }

        
        print(answer)
        return answer
