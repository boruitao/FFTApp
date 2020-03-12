#!/usr/bin/python3

import sys, getopt
import numpy as np
import matplotlib.colors
import cv2
from matplotlib.colors import LogNorm 
from matplotlib import pyplot as plt
from helper import DFT, I_DFT, FFT_2D, I_FFT, DFT_2D, FFT_2D, I_DFT_2D, I_FFT_2D

def nextPowerOf2(n): 
    count = 0
    if (n and not(n & (n - 1))): 
        return n
    while( n != 0): 
        n >>= 1
        count += 1
    return 1 << count; 

def main(argv):
    mode = ''
    imgfile = ''
    try:
        opts, args = getopt.getopt(argv,"m:i:",["md=","imgFile="])
    except getopt.GetoptError:
        print ('Error\tplease following the right syntax: python fft.py [-m mode] [-i image]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-m":
            mode = arg
        elif opt == "-i":
            imgfile = arg
    if mode == "1":
        if imgfile != '':
            img = cv2.imread(imgfile, 0)
            if img is None:
                print ('Error\tplease specify a valid path to the image')
                sys.exit(2)
            width = nextPowerOf2(img.shape[1])
            height = nextPowerOf2(img.shape[0])
            # resize the image so that the size is power of 2
            dim = (width, height)
            resized_img = cv2.resize(img, dim)
            #print(np.allclose(FFT_2D(resized_img), np.fft.fft2(resized_img)))
            f = FFT_2D(resized_img)
            plt.subplot(121), plt.imshow(resized_img, cmap = 'gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(abs(np.fft.fft2(resized_img)), norm=LogNorm())
            plt.title('Log scaled FFT'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            print ('Error\tplease specify the image which you wish to create a Fourier transform')
            sys.exit(2)
    elif mode == "2":
        print (2)
    elif mode == "3":
        print (3)
    elif mode == "4":
        print (4)
    print ('Input file is "', mode)

if __name__ == "__main__":
    main(sys.argv[1:])