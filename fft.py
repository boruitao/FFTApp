import sys, getopt
import numpy as np
import matplotlib.colors
import cv2
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from fft_method import DFT, I_DFT, FFT_2D, I_FFT, DFT_2D, FFT, I_DFT_2D, I_FFT_2D
from compress import percentile_threshold, use_high_low_freq
from runtime_exp import get_runtime_plot
def nextPower2(n):
    res = 1
    if (n and not(n & (n - 1))):
        return n
    while(res < n):
        res <<= 1
    return res

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
            width = nextPower2(img.shape[1])
            height = nextPower2(img.shape[0])
            # resize the image so that the size is power of 2
            dim = (width, height)
            resized_img = cv2.resize(img, dim)
            #print(np.allclose(FFT_2D(resized_img), np.fft.fft2(resized_img)))

            # get the Fast Fourier Transform
            f = FFT_2D(resized_img)

            # plot the 1 by 2 subplot of the original image and the FFT
            plt.subplot(121)
            plt.imshow(resized_img, cmap='gray')
            plt.title('Original Image')
            plt.subplot(122)
            #plt.plot(abs(f))
            plt.imshow(abs(f), norm=LogNorm())#, plt.colorbar()
            plt.title('Log scaled FFT')
            plt.show()
        else:
            print ('Error\tplease specify the image which you wish to create a Fourier transform')
            sys.exit(2)
    elif mode == "2":
        if imgfile != '':
            img = cv2.imread(imgfile, 0)
            if img is None:
                print ('Error\tplease specify a valid path to the image')
                sys.exit(2)
            width = nextPower2(img.shape[1])
            height = nextPower2(img.shape[0])
            # resize the image so that the size is power of 2
            dim = (width, height)
            resized_img = cv2.resize(img, dim)

            # get the Fast Fourier Transform
            f = FFT_2D(resized_img)
            
            # denoise the image based on a fraction. Any frequency wi
            fraction = 0.1
            x, y = f.shape
            nonzero_rows = int(x * fraction)
            nonzero_cols = int(y * fraction)
            f[nonzero_rows:x-nonzero_rows] = 0
            f[:, nonzero_cols:y-nonzero_cols] = 0
            total_nonzero = nonzero_rows*nonzero_cols*4
            print ('Total number of nonzeros is: ', total_nonzero, ', and the fraction is: ', round((fraction*2)*(fraction*2),3))
            
            # get the inverse Fast Fourier Transform
            denoised_img = I_FFT_2D(f)
            print(np.allclose(I_FFT_2D(f), np.fft.ifft2(f)))

            # plot the 1 by 2 subplot of the original image and the denoised image
            plt.subplot(121)
            plt.imshow(resized_img, cmap = 'gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(denoised_img.real, cmap = 'gray')
            plt.title('Denoised Image')
            plt.show()
        else:
            print ('Error\tplease specify the image which you wish to denoise')
            sys.exit(2)
    elif mode == "3":
        if imgfile != '':
            img = cv2.imread(imgfile, 0)
            if img is None:
                print ('Error\tplease specify a valid path to the image')
                sys.exit(2)
            width = nextPower2(img.shape[1])
            height = nextPower2(img.shape[0])
            # resize the image so that the size is power of 2
            dim = (width, height)
            resized_img = cv2.resize(img, dim)
            # get the Fast Fourier Transform
            f = FFT_2D(resized_img)
            #percentile_threshold(resized_img, f)
            use_high_low_freq(resized_img, f)
        else:
            print ('Error\tplease specify the image which you wish to compress')
            sys.exit(2)
    elif mode == "4":
        get_runtime_plot()
    print ('Input file is "', mode)

if __name__ == "__main__":
    main(sys.argv[1:])