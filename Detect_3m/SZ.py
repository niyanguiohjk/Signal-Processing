import numpy as np
from matplotlib import pyplot as plt 
from scipy import signal
# data = np.loadtxt('SZ1_L.csv', delimiter=',', usecols=[0, 1], unpack=True, skiprows = 1, encoding='UTF-8')
data = np.loadtxt('GZ1_R.csv', delimiter=',', usecols=[0, 1], unpack=True, encoding='UTF-8')
point = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[1], unpack=True, encoding='UTF-8')
x = data[0] 
y = data[1]
index = np.isin(x, point).nonzero()
# b, a = signal.butter(6, 0.006, 'lowpass')
# fy = signal.filtfilt(b, a, y)
plt.title('RWJ')
plt.xlabel("mileage/m")
plt.ylabel("amplitude/mm")
plt.plot(x, y)
# plt.plot(x, fy)
plt.plot(x[index], y[index], ".")
plt.grid()
plt.show()