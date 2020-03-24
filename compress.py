import numpy as np
from matplotlib import pyplot as plt
from fft_method import DFT, I_DFT, FFT_2D, I_FFT, DFT_2D, FFT, I_DFT_2D, I_FFT_2D
import scipy.sparse

def plot_images(resized_img, f1, f2, f3, f4, f5):
    # plot the 2 by 3 subplot of the original image and the compressed image
    plt.subplot(231)
    plt.imshow(resized_img, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(f1.real, cmap = 'gray')
    plt.title('19% compression')
    plt.subplot(233), plt.imshow(f2.real, cmap = 'gray')
    plt.title('38% compression')
    plt.subplot(234), plt.imshow(f3.real, cmap = 'gray')
    plt.title('57% compression')
    plt.subplot(235), plt.imshow(f4.real, cmap = 'gray')
    plt.title('76% compression')
    plt.subplot(236), plt.imshow(f5.real, cmap = 'gray')
    plt.title('95% compression')
    plt.show()

def percentile_threshold(resized_img, f):
    save_fft(f, '0%_fft.npz')

    # 19% compression
    thresh1 = np.percentile(abs(f), 19)
    f1 = (abs(f) >= thresh1) * f
    if1 = I_FFT_2D(f1)
    save_fft(f1, '19%_fft.npz')

    # 38% compression
    thresh2 = np.percentile(abs(f), 38)
    f2 = (abs(f) >= thresh2) * f
    if2 = I_FFT_2D(f2)
    save_fft(f2, '38%_fft.npz')

    # 57% compression
    thresh3 = np.percentile(abs(f), 57)
    f3 = (abs(f) >= thresh3) * f
    if3 = I_FFT_2D(f3)
    save_fft(f3, '57%_fft.npz')

    # 76% compression
    thresh4 = np.percentile(abs(f), 76)
    f4 = (abs(f) >= thresh4) * f
    if4 = I_FFT_2D(f4)
    save_fft(f4, '76%_fft.npz')

    # 95% compression
    thresh5 = np.percentile(abs(f), 95)
    f5 = (abs(f) >= thresh5) * f
    if5 = I_FFT_2D(f5)
    save_fft(f5, '95%_fft.npz')

    plot_images(resized_img, if1, if2, if3, if4, if5)

    print ("number of non zeros in 19% compression fft: ", np.count_nonzero(f1))
    print ("number of non zeros in 38% compression fft: ", np.count_nonzero(f2))
    print ("number of non zeros in 57% compression fft: ", np.count_nonzero(f3))
    print ("number of non zeros in 76% compression fft: ", np.count_nonzero(f4))
    print ("number of non zeros in 95% compression fft: ", np.count_nonzero(f5))

def high_low_freq(resized_img, f):
    # 
    fraction = 0.1
    x, y = f.shape
    nonzero_rows = int(x * fraction)
    nonzero_cols = int(y * fraction)
    f[nonzero_rows:x-nonzero_rows] = 0
    f[:, nonzero_cols:y-nonzero_cols] = 0

    # 19% compression
    thresh1 = np.percentile(abs(f), 19)
    c1 = (abs(f) >= thresh1) * f
    f1 = I_FFT_2D(c1)
    
    # 38% compression
    thresh2 = np.percentile(abs(f), 38)
    c2 = (abs(f) >= thresh2) * f
    f2 = I_FFT_2D(c2)

    # 57% compression
    thresh3 = np.percentile(abs(f), 57)
    c3 = (abs(f) >= thresh3) * f
    f3 = I_FFT_2D(c3)

    # 76% compression
    thresh4 = np.percentile(abs(f), 76)
    c4 = (abs(f) >= thresh4) * f
    f4 = I_FFT_2D(c4)

    # 95% compression
    thresh5 = np.percentile(abs(f), 95)
    c5 = (abs(f) >= thresh5) * f
    f5 = I_FFT_2D(c5)

def save_fft(f, s):
    sparse_matrix = scipy.sparse.csr_matrix(f)
    scipy.sparse.save_npz('matrix/'+s, sparse_matrix)
