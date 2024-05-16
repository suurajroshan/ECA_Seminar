# Seminar on Efficient Computational Algorithms

The _FFT.py_ file executes both DFT and FFT and times the functions to compare how the execution times of the functions may vary. It uses the output of the FFT to plot the transformed functions in the frequency domain and then passes this through a low pass filter to remove the noise. This denoised signal is inverse-transformed back to the spatial domain to get the independent sine and cosine functions. 

The _FourierSer.py_ takes the input of a function with a discrete set of points and tries to approximate the function f(x) with the sum of orthogonal sets of sine and cosine functions. The number of approximation functions can be chosen and the sampling rate of the function can be set. 
