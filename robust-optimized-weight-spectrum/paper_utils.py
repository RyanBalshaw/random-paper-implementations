from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# https://scikit-learn.org/stable/developers/develop.html#estimators
class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, param = 1):
        """
        Every keyword argument accepted by __init__ should correspond to an attribute
        on the instance. Scikit-learn relies on this to find the relevant attributes to
        set on an estimator when doing model selection. There should be no logic, not
        even input validation, and the parameters should not be changed; which also
        means ideally they should not be mutable objects such as lists or dictionaries.
        """
        self.param = param

    def _gradient(self, X, y):
        pass

    def _hessian(self, X, y):
        pass

    def decision_function(self, X):
        # Check the estimator.fit(...) method has been called.
        check_is_fitted(self)

        # Validate and process input data
        X = self._validate_data(X, reset=False)

        # Use the decision function
        # TODO

        return np.random.random(X.shape[0])

    def fit(self, X, y):
        """
        The fit method is provided on every estimator. It usually takes some samples X,
        targets y if the model is supervised, and potentially other sample properties
        such as sample_weight. It should: clear any prior attributes stored on the
        estimator, unless warm_start is used; validate and interpret any parameters,
        ideally raising an error if invalid; validate the input data; estimate and store
        model attributes from the estimated parameters and provided data; and return the
        now fitted estimator to facilitate method chaining.Target Types describes
        possible formats for y.

        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        """
        # Validate and process input data
        X, y = self._validate_data(X, y)

        # Store number of features
        self.n_features_in_ = X.shape[1]

        # Process class labels
        self.classes_, y = np.unique(y, return_inverse=True)

        # Fit estimator
        # TODO!

        # Mark estimator as fitted
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Makes a prediction for each sample, usually only taking X as input (but see
        under regressor output conventions below). In a classifier or regressor, this
        prediction is in the same target space used in fitting (e.g. one of {‘red’,
        ‘amber’, ‘green’} if the y in fitting consisted of these strings). Despite this,
        even when y passed to fit is a list or other array-like, the output of predict
        should always be an array or sparse matrix. In a clusterer or outlier detector
        the prediction is an integer.

        If the estimator was not already fitted, calling this method should raise a
        exceptions.NotFittedError.
        ---
        Output conventions:
        classifier
            An array of shape (n_samples,) (n_samples, n_outputs). Multilabel data may
            be represented as a sparse matrix if a sparse matrix was used in fitting.
            Each element should be one of the values in the classifier’s classes_
            attribute.

        """

        # Use the decision function to get the mapping of the data
        D = self.decision_function(X)

        return self.classes_[1 * (D > 0.5)]

    def predict_proba(self, X):
        """
        A method in classifiers and clusterers that can return probability estimates
        for each class/cluster. Its input is usually only some observed data, X.

        If the estimator was not already fitted, calling this method should raise a
        exceptions.NotFittedError.

        Output conventions are like those for decision_function except in the binary
        classification case, where one column is output for each class (while
        decision_function outputs a 1d array). For binary and multiclass predictions,
        each row should add to 1.

        Like other methods, predict_proba should only be present when the estimator can
        make probabilistic predictions (see duck typing). This means that the presence
        of the method may depend on estimator parameters (e.g. in
        linear_model.SGDClassifier) or training data (e.g. in
        model_selection.GridSearchCV) and may only appear after fitting.
        """
        # Check the estimator.fit(...) method has been called.
        check_is_fitted(self)

        # Validate the X data
        X = self._validate_data(X, reset=False)

        # Simple implementation for demonstration
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Create uniform probabilities for demonstration
        proba = np.ones((n_samples, n_classes)) / n_classes
        return proba


    def predict_log_proba(self, X):
        """
        The natural logarithm of the output of predict_proba, provided to facilitate
        numerical stability.
        """
        # Check the estimator.fit(...) method has been called.
        check_is_fitted(self)

        # Validate the X data
        X = self._validate_data(X, reset=False)

        # Get probabilities and compute log
        proba = self.predict_proba(X)
        return np.log(proba)

    def score(self, X, y):
        """
        A method on an estimator, usually a predictor, which evaluates its predictions
        on a given dataset, and returns a single numerical score. A greater return value
        should indicate better predictions; accuracy is used for classifiers and R^2
        for regressors by default. If the estimator was not already fitted, calling
        this method should raise a exceptions.NotFittedError. Some estimators implement
        a custom, estimator-specific score function, often the likelihood of the data
        under the model.
        """

        # Validate the X data
        X = self._validate_data(X, reset=False)

        y_pred = self.predict(X)

        return np.mean(y_pred == y)

    def score_samples(self, X):
        """
        A method that returns a score for each given sample. The exact definition of
        score varies from one class to another. In the case of density estimation, it
        can be the log density model on the data, and in the case of outlier detection,
        it can be the opposite of the outlier factor of the data. If the estimator was
        not already fitted, calling this method should raise an
        exceptions.NotFittedError.
        """
        pass

        # Validate the X data
        X = self._validate_data(X, reset=False)

        return self.decision_function(X)

if __name__ == "__main__":
    check_estimator(LogisticRegression())