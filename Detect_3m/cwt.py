import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
data = np.loadtxt('dataK034717_1.csv', delimiter=',', usecols=[0, 1], unpack=True)
x = data[0] 
y = data[1]
N = len(y)

# 定义小波函数和宽度范围
wavelet = signal.morlet  
widths = np.arange(1, 501)  # 定义小波变换的宽度范围

# 执行连续小波变换
cwtmatr = signal.cwt(y, wavelet, widths)
print(cwtmatr.shape)

plt.subplot(211)
plt.imshow(cwtmatr,  cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.colorbar()
plt.title('Continuous Wavelet Transform (CWT) of the signal')
# plt.plot(y)
plt.ylabel('Scale (width)')
plt.xlabel('Time')
plt.subplot(212)
plt.title("TD")
plt.xlabel("mileage/m")
plt.ylabel("amplitude/mm")
plt.plot(x, y)
plt.grid()
plt.show()