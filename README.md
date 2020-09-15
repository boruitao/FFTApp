# Fast Fourier Transform and Applications

This is the python project to implement 2D fast fourier transform from scratch, which is used to compress and denoise images. The program can be invoked from the command line using the following syntax: 
```
python fft.py [-m mode] [-i image]
```
where the arguments are defined as follows:
* mode
  * 1: for fast mode where the image is converted into its FFT form and displayed.
  * 2: for denoising where the image is denoised by applying an FFT, truncating high
frequencies and then displayed.
  * 3: for compressing and saving the image.
  * 4: for plotting the runtime graphs.
* image (optional): filename of the image we wish to take the DFT of. 


#### Example output for each mode: 
Mode 1: the image is converted into its FFT and displayed
![Image of mode1](https://github.com/boruitao/FFT/blob/master/images/mode1.png)

Mode 2: the image is denoised
![Image of mode2](https://github.com/boruitao/FFT/blob/master/images/mode2.png)

Mode 3: the image compressed using different schemes
![Image of mode3](https://github.com/boruitao/FFT/blob/master/images/mode3.png)
