import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def DFT(input):
    input = np.asarray(input, dtype=np.float32)
    N = input.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    L = np.exp(-2j * np.pi * k * n/N)
    return np.dot(L, input)

def FFT(input):
    N = len(input)
    if N == 1:
        return input
    else:
        assert N % 2 == 0
        Feven = FFT(input[::2])
        Fodd = FFT(input[1::2])
        dk_factor = np.exp(-2j * np.pi * np.arange(N) / N)
        c = Feven
        comb = np.concatenate([c+dk_factor[:int(N/2)]*Fodd, c+dk_factor[int(N/2):]*Fodd])
    return comb

def IFFT(input):
    N = len(input)
    if N == 1:
        return input
    else:
        assert N % 2 == 0
        input_conj = np.conj(input)
        ifft = FFT(input_conj)
        out = np.conj(ifft) / N
    return out


sampling_rate = 1024
sampling_int = 1/sampling_rate
t = np.arange(0,1,sampling_int)

# generating the signal by adding frequencies of 3 independent frequencies
f = 20
x = np.sin(2*np.pi*f*t)
f = 50
x += np.sin(2*np.pi*f*t)
x_clean = x
noise = np.random.randn(len(x))
x = x + noise

plt.figure(figsize=(8,6))
plt.plot(t, x_clean, label='clean', color='k', linewidth=2)
plt.plot(t, x, label='noisy', color = 'mediumvioletred', linewidth = 2)
plt.xlim(t[0], t[-1])
plt.show()

start = timer()
xdft= DFT(x)
end = timer()
print("The execution time with DFT was: ", (end-start))

start = timer()
xhat= FFT(x)
end = timer()
print("The execution time with FFT was: ", (end-start))
psd = xhat * np.conj(xhat)/ sampling_rate
freq = t * sampling_rate
L = np.arange(1, np.floor(sampling_rate/2), dtype=int)


fig, axs = plt.subplots(3,1)

plt.sca(axs[0])
plt.plot(t, x, color = 'c', linewidth = 1.5, label = 'Noisy')
plt.plot(t, x_clean, color = 'k', linewidth = 2, label = 'Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], psd[L], color = 'c', linewidth = 1.5, label = 'Noisy')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

idx = psd > 100
psd_c = psd*idx
xhat = idx*xhat
x_c = IFFT(xhat)

plt.sca(axs[2])
plt.plot(t, x_c, color = 'c', linewidth = 1.5, label = 'Filtered')
plt.xlim(t[0], t[-1])
plt.legend()
plt.savefig('fft.png')
plt.show()

