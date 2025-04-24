import numpy as np
import scipy


def fourier_spectrum(signal):
    N = len(signal)
    return 1 / N * np.abs(np.fft.fft(signal))


def normalised_fourier_spectrum(signal):
    X = fourier_spectrum(signal)
    return X / np.sum(X)


def square_envelope(signal):
    analytic_signal = scipy.signal.hilbert(signal)
    return np.abs(analytic_signal) ** 2


def normalised_square_envelope(signal):
    SE = square_envelope(signal)
    return SE / np.sum(SE)


def square_envelope_spectrum(signal):
    SE = square_envelope(signal)

    N = len(SE)
    return (1 / N * np.abs(np.fft.fft(SE))) ** 2


def normalised_square_envelope_spectrum(signal):
    SES = square_envelope_spectrum(signal)
    return SES / np.sum(SES)
