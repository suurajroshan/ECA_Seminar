# Author: Suuraj Perpeli
# Code inspired from Brunton & Kutz (Data-Driven Science and Engineering)
# func.py only works for symmetric domain functions


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy import signal

# Defining the domain
dx = 0.005
L = 2*np.pi
x = np.arange(0, 1, dx)
n = len(x)
nquart = int(np.floor(n/4))

y = signal.sawtooth(L * x)
x_c = []
x_c = np.array([np.concatenate((x_c, x+i)) for i in range(3)]).reshape(-1,)
y_c = np.tile(y,3)
fig, ax = plt.subplots()
ax.plot(x_c, y_c,'-', color='k', linewidth=2, label="Original")

name = 'Accent'
cmap = colormaps.get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)

a0 = np.sum(y_c * np.ones_like(x_c)) * dx
fFs = a0/2

appx_num = 100
A = np.zeros(appx_num)
B = np.zeros(appx_num)
for k in range(appx_num):
    A[k] = np.sum(y_c * np.cos(np.pi*(k+1)*x_c/L)) * dx
    B[k] = np.sum(y_c * np.sin(np.pi*(k+1)*x_c/L)) * dx
    fFs = fFs + (1/L)*(A[k]*np.cos((k+1)*np.pi*x_c/L) + B[k]*np.sin((k+1)*np.pi*x_c/L))
    if k<20: ax.plot(x_c, fFs, '-')
# ax.plot(x_c, fFs, '-',color="red", label="Fourier Series Sum")
plt.xlabel("x")
plt.ylabel("Amplitude")
# ax.legend()

# plt.savefig('FourierSeries2.png')
plt.show()
