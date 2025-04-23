import os

from matplotlib import pyplot as plt
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import validate_data
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.datasets import make_circles, make_moons, make_classification, make_blobs

# https://scikit-learn.org/stable/developers/develop.html#estimators
class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, max_iter=100):
        """
        Every keyword argument accepted by __init__ should correspond to an attribute
        on the instance. Scikit-learn relies on this to find the relevant attributes to
        set on an estimator when doing model selection. There should be no logic, not
        even input validation, and the parameters should not be changed; which also
        means ideally they should not be mutable objects such as lists or dictionaries.
        """
        self.max_iter =max_iter

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

    def _loss_function(self, X, y):

        s = self.decision_function(X)

        y_vec = y.reshape(-1, 1)

        ll_vec = y_vec * np.log(s) + (1 - y_vec) * np.log(1 - s)

        return -np.mean(ll_vec) + 0.1 * np.linalg.norm(self._zeta)

    def _gradient(self, X, y):

        X_expanded = self.expand_X(X)
        s = self.decision_function(X)

        return 1/self.n_samples_ * X_expanded.T @ (s - y.reshape(-1, 1)) + 0.001 * self._zeta

    def _hessian(self, X, y):
        X_expanded = self.expand_X(X)
        s = self.decision_function(X)
        sigma_grad = s * (1 - s)
        inner_term = (sigma_grad * (1 - y.reshape(-1, 1))).T
        return (1/self.n_samples_ * X_expanded.T * inner_term) @ X_expanded + 0.001

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

        # Initialise the coefficients
        self._initialise_coefficients()

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
        X = validate_data(X, reset=False)

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
        X = validate_data(self, X, reset=False)

        return self.decision_function(X)

if __name__ == "__main__":
    # TODO: Run this
    # check_estimator(LogisticRegression())

    # X, y = make_classification(n_features = 2, n_informative = 2, n_redundant=0)
    # X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5)
    # X, y = make_circles(n_samples=1000,noise=0.01, random_state=0)
    base_dir = "../Datasets/IMS/dataset_two/"
    file_list = sorted(os.listdir(base_dir))

    f1 = np.loadtxt(os.path.join(base_dir, file_list[0]))
    f2 = np.loadtxt(os.path.join(base_dir, file_list[900]))

    N = f1.shape[0]
    Fs = 20480
    X1 =  2/N *np.abs(np.fft.fft(f1[:, 0]))[:N//2]
    X2 = 2/N * np.abs(np.fft.fft(f2[:, 0]))[:N//2]
    freq = np.fft.fftfreq(N, 1/Fs)[:N//2]
    X = np.vstack([X1.reshape(1, -1), X2.reshape(1, -1)])
    y = np.array([0, 1])


    print(X.shape, y.shape)

    plt.figure()
    plt.plot(freq, X2)
    plt.plot(freq, X1)
    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    LR_inst = LogisticRegression(100)
    LR_inst.fit(X, y)
    lr_loss = []
    lr_grad = LR_inst._gradient(X, y)
    print(lr_grad.shape)

    for i in range(1000):
        lr_loss.append(LR_inst._loss_function(X, y))
        lr_grad = LR_inst._gradient(X, y)
        # lr_hess = LR_inst._hessian(X, y)

        LR_inst._update_coefficients(-0.5 * lr_grad)  # np.linalg.solve(lr_hess, lr_grad)

    plt.figure()
    plt.plot(lr_loss)
    plt.show()

    # X_grid, Y_grid = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    # X_test = np.hstack([X_grid.ravel().reshape(-1, 1), Y_grid.ravel().reshape(-1, 1)])
    #
    # print(X_test.shape)
    # D = LR_inst.score_samples(X_test)

    # plt.figure()
    # plt.contourf(X_grid, Y_grid, D.reshape(100, 100), cmap=plt.cm.jet)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    plt.figure()
    plt.plot(np.arange(len(LR_inst._zeta[:, 0])), LR_inst._zeta[:, 0], lw = 0.4)
    plt.show()