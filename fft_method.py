import numpy as np
import cv2

# We calculate DFT using the format: X = E * x, where X and x are two vector of size N.
# Therefore, we need to compute an N*N matrix which contains the coefficient of the 
# form: exp(-2pi*i*k*n/N). 
def DFT(x):
    N = len(x)
    # n is 1xN and k is Nx1, which are used to compute E
    n = np.arange(N)
    k = n.reshape((N, 1))
    E = np.exp(-2j * np.pi * k * n / N)
    return np.dot(E, x)

# We use the exact same technique to compute the Inverse of DFT, except that we use 
# exp(2pi*i*k*n/N) instead of exp(-2pi*i*k*n/N).
def I_DFT_unscaled(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    E = np.exp(2j * np.pi * k * n / N)
    return np.dot(E, x)

def FFT(x):
    N = len(x)
    if N <= 16:
        return DFT(x)
    else:
        # Recursive step: 
        # 1. Take the FFT of even part and the odd part of x vector
        # 2. Compute the multiplier matrix e^(-2i * pi * k / N)
        # 3. Sum the odd part and the even part by concatenating
        #    even + (odd * first half of the multiplier)
        #    and 
        #    even + (odd * second half of the multiplier)
        # The whole process can be interpreted as Xk = even[k] + odd[k]*exp(-2i*pi*k/N)
        k = np.arange(N)
        ei = np.exp(-2j * np.pi * k / N)
        even = FFT([x[i] for i in range(0, N, 2)])
        odd = FFT([x[i] for i in range(1, N, 2)])
        half = N // 2
        return np.concatenate([even + ei[:half] * odd, even + ei[half:] * odd])
    
def I_FFT_unscaled(x):
    N = len(x)
    if N <= 16:
        return I_DFT_unscaled(x)
    else:
        k = np.arange(N)
        ei = np.exp(2j * np.pi * k / N)
        even = I_FFT_unscaled([x[i] for i in range(0, N, 2)])
        odd = I_FFT_unscaled([x[i] for i in range(1, N, 2)])
        half = N // 2
        return np.concatenate([even + ei[:half] * odd, even + ei[half:] * odd])

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

# First, compute the Fourier Transform (FT) for each row of the input matrix
# Then, compute the FT column-wise. This is done by taking the transpose of the 
# resulting matrix in the first step and apply FT. 
# Once we are done, take the transpose of the resulting matrix as the output.
def FT_2D(matrix, FT):
    rows = np.array([FT(row) for row in matrix])
    return np.array([FT(row) for row in rows.transpose()]).transpose()

def I_FT_2D(matrix, IFT_unscaled):
    N, M = matrix.shape
    rows = np.array([IFT_unscaled(row) for row in matrix])
    return np.array([IFT_unscaled(row) for row in rows.transpose()]).transpose() / N / M