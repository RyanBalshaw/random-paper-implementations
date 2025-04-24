from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from abc import ABC, abstractmethod
from enum import Enum

class RegType(Enum):
    L1 = "L1"
    L2 = "L2"
    ELASTIC = "ELASTIC"

class BaseRegulariser(ABC):

    def __init__(self, alpha: float):
        self.alpha = alpha

    @abstractmethod
    def loss(self, coefficients):
        pass

    @abstractmethod
    def gradient(self, coefficients):
        pass

    @abstractmethod
    def hessian(self, coefficients):
        pass

class L1Regulariser(BaseRegulariser):
    def loss(self, coefficients):
        # alpha * |w|
        return self.alpha * np.sum(np.abs(coefficients))

    def gradient(self, coefficients):
        # alpha * sign(w)
        return self.alpha * np.sign(coefficients)

    def hessian(self, coefficients):
        # L1 regularization doesn't have a true Hessian (non-differentiable at 0)
        # But for practical purposes:
        return 0.0001 * np.eye(len(coefficients))  # Small constant for numerical stability


class L2Regulariser(BaseRegulariser):
    def loss(self, coefficients):
        # alpha * w^Tw
        return self.alpha * np.sum(coefficients ** 2)

    def gradient(self, coefficients):
        # alpha * 2 * w
        return self.alpha * 2 * coefficients

    def hessian(self, coefficients):
        # alpha * 2 * I
        return self.alpha * 2 * np.eye(len(coefficients))


class ElasticNetRegulariser(BaseRegulariser):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super().__init__(alpha)
        self.l1_ratio = l1_ratio
        self._l1 = L1Regulariser(alpha)
        self._l2 = L2Regulariser(alpha)

        assert 0 <= self.l1_ratio <= 1, "l1_ratio must be between 0 and 1."

    def loss(self, coefficients):
        l1 = self._l1.loss(coefficients)
        l2 = self._l2.loss(coefficients)
        return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2

    def gradient(self, coefficients):
        l1 = self._l1.gradient(coefficients)
        l2 = self._l2.gradient(coefficients)
        return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2

    def hessian(self, coefficients):
        l1 = self._l1.hessian(coefficients)
        l2 = self._l2.hessian(coefficients)
        return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2

# https://scikit-learn.org/stable/developers/develop.html#estimators
class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, learning_rate: float = 0.01, max_iter:int=100, regulariser_type: Optional[RegType] = None, alpha: Optional[float] = None, l1_ratio: Optional[float] = None):
        """
        Every keyword argument accepted by __init__ should correspond to an attribute
        on the instance. Scikit-learn relies on this to find the relevant attributes to
        set on an estimator when doing model selection. There should be no logic, not
        even input validation, and the parameters should not be changed; which also
        means ideally they should not be mutable objects such as lists or dictionaries.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regulariser_type = regulariser_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _get_regulariser(self):

        if self.regulariser_type is None:
            self.regulariser = None
            return
        else:
            if self.regulariser_type == RegType.L1:
                self.regulariser = L1Regulariser(self.alpha)

            elif self.regulariser_type == RegType.L2:
                self.regulariser = L2Regulariser(self.alpha)

            elif self.regulariser_type == RegType.ELASTIC:
                self.regulariser = ElasticNetRegulariser(self.alpha, self.l1_ratio)
            else:
                raise ValueError(f"Unknown regulariser type: {self.regulariser_type}")

    def _initialise_coefficients(self):
        self._zeta = np.random.randn(self.n_features_in_ + 1, 1) * 0.1

    def _sigmoid(self, u):
        return 1 / (1 + np.exp(-u))

    def expand_X(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _z_mapping(self, X):
        return self.expand_X(X) @ self._zeta

    def decision_function(self, X):
        # Check the estimator.fit(...) method has been called.
        check_is_fitted(self)

        # Validate and process input data
        X = validate_data(self, X, reset=False)

        # Use the decision function
        z = self._z_mapping(X)
        s = self._sigmoid(z)

        return s

    def _loss_function(self, X, y, aggregate: bool = True):
        s = self.decision_function(X)

        y_vec = y.reshape(-1, 1)

        ll_vec = -1 * (y_vec * np.log(s) + (1 - y_vec) * np.log(1 - s))

        if aggregate:
            loss = np.mean(ll_vec)
        else:
            loss = ll_vec

        if self.regulariser and not aggregate:
            reg_loss = self.regulariser.loss(self._zeta)
            return loss + reg_loss

        return loss

    def _gradient(self, X, y):
        X_expanded = self.expand_X(X)
        s = self.decision_function(X)

        grad = 1 / self.n_samples_ * X_expanded.T @ (
            s - y.reshape(-1, 1)
        )

        if self.regulariser:
            reg_grad = self.regulariser.gradient(self._zeta)
            return grad + reg_grad

        return grad

    def _hessian(self, X, y):
        X_expanded = self.expand_X(X)
        s = self.decision_function(X)
        sigma_grad = s * (1 - s)
        inner_term = (sigma_grad * (1 - y.reshape(-1, 1))).T

        hess =  (1 / self.n_samples_ * X_expanded.T * inner_term) @ X_expanded

        if self.regulariser:
            reg_hess = self.regulariser.hessian(self._zeta)
            return hess + reg_hess

        return hess

    def _update_coefficients(self, delta_coefficients):
        self._zeta += delta_coefficients

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
        # Store number of features and number of samples
        self.n_samples_, self.n_features_in_ = X.shape

        # Validate and process input data
        X, y = validate_data(self, X, y)

        # Set up regularizer
        self._get_regulariser()

        # Initialise the coefficients
        self._initialise_coefficients()

        # Process class labels
        self.classes_, y = np.unique(y, return_inverse=True)

        # Fit estimator using gradient descent
        # TODO: Extend to different optimisiation approaches

        self._loss_values = []
        for _ in range(self.max_iter):

            # Calculate loss
            loss = self._loss_function(X, y)
            self._loss_values.append(loss)

            # Calculate gradient
            grad = self._gradient(X, y)

            # Update coefficients
            delta_zeta = -1 * self.learning_rate * grad
            self._update_coefficients(delta_zeta)

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
        X = validate_data(X, reset=False)

        # Simple implementation for demonstration
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Create uniform probabilities for demonstration
        proba = self.decision_function(X)
        return proba

    def predict_log_proba(self, X):
        """
        The natural logarithm of the output of predict_proba, provided to facilitate
        numerical stability.
        """
        # Check the estimator.fit(...) method has been called.
        check_is_fitted(self)

        # Validate the X data
        X = validate_data(X, reset=False)

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
        X = validate_data(X, reset=False)

        y_pred = self.predict(X)

        return np.mean(y_pred == y)
