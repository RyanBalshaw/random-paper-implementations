import os

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # TODO: Run this
    # check_estimator(LogisticRegression())

    # X, y = make_classification(n_features = 2, n_informative = 2, n_redundant=0)
    # X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5)
    # X, y = make_circles(n_samples=1000,noise=0.01, random_state=0)
    base_dir = "../Datasets/IMS/dataset_two/"
    file_list = sorted(os.listdir(base_dir))

    f1 = np.loadtxt(os.path.join(base_dir, file_list[600]))
    f2 = np.loadtxt(os.path.join(base_dir, file_list[700]))

    N = f1.shape[0]
    Fs = 20480
    X1 = 2 / N * np.abs(np.fft.fft(f1[:, 0]))[: N // 2]
    X2 = 2 / N * np.abs(np.fft.fft(f2[:, 0]))[: N // 2]
    freq = np.fft.fftfreq(N, 1 / Fs)[: N // 2]
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

    for i in range(500):
        lr_loss.append(LR_inst._loss_function(X, y))
        lr_grad = LR_inst._gradient(X, y)
        # lr_hess = LR_inst._hessian(X, y)

        LR_inst._update_coefficients(
            -0.5 * lr_grad
        )  # np.linalg.solve(lr_hess, lr_grad)

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
    plt.plot(np.arange(len(LR_inst._zeta[:, 0])), LR_inst._zeta[:, 0], lw=0.4)
    plt.show()
