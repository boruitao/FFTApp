import numpy as np
import cv2

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def I_DFT_unscaled(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    N = len(x)
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return DFT(x)
    else:
        X_even = FFT([x[i] for i in range(0, N, 2)])
        X_odd = FFT([x[i] for i in range(1, N, 2)])
        
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd])
    
def I_FFT_unscaled(x):
    N = len(x)
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return I_DFT_unscaled(x)
    else:
        X_even = I_FFT_unscaled(x[::2])
        X_odd = I_FFT_unscaled(x[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd])

def I_FFT(x):
    N = len(x)
    return I_FFT_unscaled(x) / N

def I_DFT(x):
    N = len(x)
    return I_DFT_unscaled(x) / N

def DFT_2D(matrix):
    fftRows = np.array([DFT(row) for row in matrix])
    return np.array([DFT(row) for row in fftRows.transpose()]).transpose()

def FFT_2D(matrix):
    fftRows = np.array([FFT(row) for row in matrix])
    return np.array([FFT(row) for row in fftRows.transpose()]).transpose()

def I_DFT_2D(matrix):
    N, M = matrix.shape
    fftRows = np.array([I_DFT_unscaled(row) for row in matrix])
    return np.array([I_DFT_unscaled(row) for row in fftRows.transpose()]).transpose() / N / M

def I_FFT_2D(matrix):
    N, M = matrix.shape
    fftRows = np.array([I_FFT_unscaled(row) for row in matrix])
    return np.array([I_FFT_unscaled(row) for row in fftRows.transpose()]).transpose() / N / M