from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

EPSILON = 1e-10


class RegType(Enum):
    L1 = "L1"
    L2 = "L2"
    ELASTIC = "ELASTIC"


class BaseRegulariser(ABC):
    def __init__(self, alpha: float):
        self.alpha = alpha
        assert self.alpha >= 0, "alpha must be non-negative."

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
        return 0.0001 * np.eye(
            len(coefficients)
        )  # Small constant for numerical stability


class L2Regulariser(BaseRegulariser):
    def loss(self, coefficients):
        # alpha * w^Tw
        return self.alpha * np.sum(coefficients**2)

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


class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        regulariser_type: Optional[RegType] = None,
        alpha: Optional[float] = None,
        l1_ratio: Optional[float] = None,
        optimiser: str = "gradient_descent",
        tol: float = 1e-4,
    ):
        """
        https://scikit-learn.org/stable/developers/develop.html#estimators

        Parameters:
        -----------
        learning_rate : float, default=0.01
            Step size for gradient descent

        max_iter : int, default=100
            Maximum number of iterations for optimisation

        regulariser_type : RegType, default=None
            Type of regularisation to use

        alpha : float, default=None
            Regularisation strength

        l1_ratio : float, default=None
            Ratio of L1 regularisation for ElasticNet

        optimiser : str, default='gradient_descent'
            Optimisation algorithm to use: 'gradient_descent', 'bfgs', 'l-bfgs-b',
            'newton-cg', etc.

        tol : float, default=1e-4
            Tolerance for stopping criterion
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regulariser_type = regulariser_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.optimiser = optimiser
        self.tol = tol

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
        self._zeta = np.random.randn(self.n_features_in_ + 1, 1) * 0.01

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

        ll_vec = -1 * (
            y_vec * np.log(s + EPSILON) + (1 - y_vec) * np.log(1 - s + EPSILON)
        )

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

        grad = 1 / self.n_samples_ * X_expanded.T @ (s - y.reshape(-1, 1))

        if self.regulariser:
            reg_grad = self.regulariser.gradient(self._zeta)
            return grad + reg_grad

        return grad

    def _hessian(self, X, y):
        X_expanded = self.expand_X(X)
        s = self.decision_function(X)
        sigma_grad = s * (1 - s)
        weighted_X = sigma_grad * X_expanded

        hess = (1 / self.n_samples_) * X_expanded.T @ weighted_X

        if self.regulariser:
            reg_hess = self.regulariser.hessian(self._zeta)
            return hess + reg_hess

        return hess

    def _update_coefficients(self, delta_coefficients):
        self._zeta += delta_coefficients

    def _objective_function(self, coefficients, X, y):
        """Objective function for scipy.optimize.minimize"""
        # Reshape the coefficients to the expected shape
        self._zeta = coefficients.reshape(-1, 1)
        return self._loss_function(X, y)

    def _objective_gradient(self, coefficients, X, y):
        """Gradient of the objective function for scipy.optimize.minimize"""
        # Reshape the coefficients to the expected shape
        self._zeta = coefficients.reshape(-1, 1)
        return self._gradient(X, y).flatten()

    def _objective_hessian(self, coefficients, X, y):
        """Gradient of the objective function for scipy.optimize.minimize"""
        # Reshape the coefficients to the expected shape
        self._zeta = coefficients.reshape(-1, 1)
        return self._hessian(X, y)

    def _optimise_coefficients(self, X, y):
        if self.optimiser.lower() == "gradient_descent":
            # Standard gradient descent
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
        else:
            # Use scipy.optimize.minimize
            initial_coeffs = self._zeta.flatten()  # Flatten for scipy.optimize
            method_args = (X, y)

            if self.optimiser.lower() in ["bfgs", "l-bfgs-b", "newton-cg"]:
                options = {"maxiter": self.max_iter, "disp": False}

                # For methods that support Hessian
                if self.optimiser.lower() == "newton-cg":
                    result = minimize(
                        fun=self._objective_function,
                        x0=initial_coeffs,
                        args=method_args,
                        method=self.optimiser,
                        jac=self._objective_gradient,
                        hess=self._objective_hessian,
                        options=options,
                    )
                else:
                    result = minimize(
                        fun=self._objective_function,
                        x0=initial_coeffs,
                        args=method_args,
                        method=self.optimiser,
                        jac=self._objective_gradient,
                        options=options,
                    )
            else:
                # For methods that don't need gradient
                options = {"maxiter": self.max_iter, "disp": False}
                result = minimize(
                    fun=self._objective_function,
                    x0=initial_coeffs,
                    args=method_args,
                    method=self.optimiser,
                    options=options,
                )

            # Store optimization results
            self._optimization_result = result
            self._zeta = result.x.reshape(-1, 1)
            self._loss_values = [result.fun]  # Just the final loss value

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
        self._optimise_coefficients(X, y)

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
        X = validate_data(self, X, reset=False)

        # Get probabilities
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
        X = validate_data(self, X, reset=False)

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

    def check_gradient_finite_diff(self, X, y, step_size=1e-6):
        """
        Checks the gradient implementation using finite differences.

        Parameters
        ----------
        X : ndarray
            Input features of shape (n_samples, n_features)
        y : ndarray
            Target values of shape (n_samples,)
        step_size : float, default=1e-6
            Step size for finite difference approximation

        Returns
        -------
        analytical_grad : ndarray
            Gradient computed using the analytical formula
        numerical_grad : ndarray
            Gradient computed using finite differences
        rel_error : float
            Relative error between analytical and numerical gradients
        """
        # Store current coefficients
        original_zeta = self._zeta.copy()

        # Calculate analytical gradient
        analytical_grad = self._gradient(X, y)

        # Calculate numerical gradient
        numerical_grad = np.zeros_like(self._zeta)
        for i in range(len(self._zeta)):
            # Forward step
            self._zeta = original_zeta.copy()
            self._zeta[i] += step_size
            loss_plus = self._loss_function(X, y)

            # Backward step
            self._zeta = original_zeta.copy()
            self._zeta[i] -= step_size
            loss_minus = self._loss_function(X, y)

            # Central difference approximation
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * step_size)

        # Restore original coefficients
        self._zeta = original_zeta

        # Calculate relative error
        abs_diff = np.linalg.norm(analytical_grad - numerical_grad)
        abs_grad = np.linalg.norm(analytical_grad)
        rel_error = abs_diff / (abs_grad + EPSILON)

        return analytical_grad, numerical_grad, rel_error

    def check_hessian_finite_diff(self, X, y, step_size=1e-6):
        """
        Checks the Hessian implementation using finite differences.

        Parameters
        ----------
        X : ndarray
            Input features of shape (n_samples, n_features)
        y : ndarray
            Target values of shape (n_samples,)
        step_size : float, default=1e-6
            Step size for finite difference approximation

        Returns
        -------
        analytical_hess : ndarray
            Hessian computed using the analytical formula
        numerical_hess : ndarray
            Hessian computed using finite differences
        rel_error : float
            Relative error between analytical and numerical Hessians
        """
        # Store current coefficients
        original_zeta = self._zeta.copy()

        # Calculate analytical Hessian
        analytical_hess = self._hessian(X, y)

        # Calculate numerical Hessian using finite differences on the gradient
        n_params = len(self._zeta)
        numerical_hess = np.zeros((n_params, n_params))

        for i in range(n_params):
            for j in range(n_params):
                # Forward step for parameter i and j
                self._zeta = original_zeta.copy()
                self._zeta[i] += step_size
                self._zeta[j] += step_size
                f1 = self._loss_function(X, y)

                # Backward step for parameter i and j
                self._zeta = original_zeta.copy()
                self._zeta[i] -= step_size
                self._zeta[j] -= step_size
                f2 = self._loss_function(X, y)

                # Symmetrical step for parameter i and j
                self._zeta = original_zeta.copy()
                self._zeta[i] += step_size
                self._zeta[j] -= step_size
                f3 = self._loss_function(X, y)

                # Symmetrical step for parameter i and j
                self._zeta = original_zeta.copy()
                self._zeta[i] -= step_size
                self._zeta[j] += step_size
                f4 = self._loss_function(X, y)

                # Compute the i-th column of the Hessian using central difference
                numerical_hess[i, j] = (f1 + f2 - f3 - f4) / (4 * step_size**2)

        # Restore original coefficients
        self._zeta = original_zeta

        # Calculate relative error
        abs_diff = np.linalg.norm(analytical_hess - numerical_hess)
        abs_hess = np.linalg.norm(analytical_hess)
        rel_error = abs_diff / (abs_hess + EPSILON)

        return analytical_hess, numerical_hess, rel_error

    def check_gradients_and_hessian(self, X, y, step_size=1e-6, verbose=True):
        """
        Performs gradient and Hessian verification using finite differences.

        Parameters
        ----------
        X : ndarray
            Input features of shape (n_samples, n_features)
        y : ndarray
            Target values of shape (n_samples,)
        step_size : float, default=1e-6
            Step size for finite difference approximation
        verbose : bool, default=True
            Whether to print results

        Returns
        -------
        grad_err : float
            Relative error for gradient
        hess_err : float
            Relative error for Hessian
        """
        # Check the gradient
        _, _, grad_err = self.check_gradient_finite_diff(X, y, step_size)

        if verbose:
            print(f"Gradient verification - relative error: {grad_err:.6e}")

            if grad_err < 1e-5:
                print("Gradient implementation appears correct!")
            else:
                print("Gradient implementation might have issues!")

        # Check the Hessian
        _, _, hess_err = self.check_hessian_finite_diff(X, y, step_size)

        if verbose:
            print(f"Hessian verification - relative error: {hess_err:.6e}")

            if hess_err < 1e-3:
                print("Hessian implementation appears correct!")
            else:
                print("Hessian implementation might have issues!")

        return grad_err, hess_err
