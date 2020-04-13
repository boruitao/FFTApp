import time
import numpy as np
from fft_method import DFT, I_DFT, FFT_2D, I_FFT, DFT_2D, FFT, I_DFT_2D, I_FFT_2D
import matplotlib.pyplot as plt

def get_single_runtime(x, y, ft2d):
    img = np.random.rand(x, y)
    start_time = time.time()
    f = ft2d(img)
    runtime = time.time() - start_time
    return runtime

def get_mean_std(x, y, ft2d):
    exps = []
    for i in range(10):
        exps.append(get_single_runtime(x, y, ft2d))
    return np.mean(exps), np.std(exps)

def get_means_stds(problem_sizes, ft2d):
    means = []
    stds = []
    for i in problem_sizes:
        x = int(i / 2)
        y = i-x
        mean, std = get_mean_std(2**x, 2**y, ft2d)
        means.append(mean)
        stds.append(std)
    return means, stds

def get_runtime_plot():
    problem_sizes = [10,11,12,13,14,15,16,17,18]
    print('problem sizes (2 to the power):', problem_sizes)
    print()

    means_FFT, stds_FFT = get_means_stds(problem_sizes, FFT_2D)
    print('FFT:')
    print('mean', means_FFT)
    print('standard deviation', stds_FFT)
    print()

    means_DFT, stds_DFT = get_means_stds(problem_sizes, DFT_2D)
    print('naive method:')
    print('mean', means_DFT)
    print('standard deviation', stds_DFT)

    plt.errorbar(problem_sizes, means_FFT, yerr=stds_FFT, uplims=True, lolims=True, label='FFT')
    plt.errorbar(problem_sizes, means_DFT, yerr=stds_DFT, uplims=True, lolims=True, label='naive method')

    plt.xlabel('problem size [2 to the power]')
    plt.ylabel('mean run time [s]')
    plt.legend(loc='upper left')
    plt.show()
