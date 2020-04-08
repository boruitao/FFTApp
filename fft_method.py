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
    if N <= 32:
        return DFT(x)
    else:
        ei = np.exp(-2j * np.pi * np.arange(N) / N)
        even = FFT([x[i] for i in range(0, N, 2)])
        odd = FFT([x[i] for i in range(1, N, 2)])
        return np.concatenate([even + ei[:N // 2] * odd, even + ei[N // 2:] * odd])
    
def I_FFT_unscaled(x):
    N = len(x)
    if N <= 32:
        return I_DFT_unscaled(x)
    else:
        ei = np.exp(2j * np.pi * np.arange(N) / N)
        even = I_FFT_unscaled(x[::2])
        odd = I_FFT_unscaled(x[1::2])
        return np.concatenate([even + ei[:N // 2] * odd, even + ei[N // 2:] * odd])

def I_FFT(x):
    N = len(x)
    return I_FFT_unscaled(x) / N

def I_DFT(x):
    N = len(x)
    return I_DFT_unscaled(x) / N

def DFT_2D(matrix):
    return FT_2D(matrix, DFT)

def FFT_2D(matrix):
    return FT_2D(matrix, FFT)

def I_DFT_2D(matrix):
    return I_FT_2D(matrix, I_DFT_unscaled)

def I_FFT_2D(matrix):
    return I_FT_2D(matrix, I_FFT_unscaled)

def FT_2D(matrix, FT):
    rows = np.array([FT(row) for row in matrix])
    return np.array([FT(row) for row in rows.transpose()]).transpose()

def I_FT_2D(matrix, IFT_unscaled):
    N, M = matrix.shape
    rows = np.array([IFT_unscaled(row) for row in matrix])
    return np.array([IFT_unscaled(row) for row in rows.transpose()]).transpose() / N / M