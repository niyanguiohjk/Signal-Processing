import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import DataLabel
# print(Dataset.dataset[0], Dataset.label[0])
# FFT
# y_fft = np.abs(np.fft.fft(Dataset.dataset[0])[:100000//2]) / 100000 * 2
# freq = np.fft.fftfreq(100000, 0.001)[:100000//2]
# plt.plot(freq, y_fft)
# plt.show() 

fs = 1000  # 采样频率
t = np.linspace(0, 100, fs, endpoint=False)  # 时间轴
x = DataLabel.dataset[0]  

# 执行短时傅里叶变换
f, t, Zxx = signal.stft(x, fs, window='hann', nperseg=512)  # nperseg是每个段的长度

# 绘制时频图
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.01, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Mileage [m]')
plt.colorbar(label='Magnitude')
plt.ylim(5, 10)
for index, value in enumerate(DataLabel.label[0]):
    if value == 1:
        start = 4*index
        end = 4*index + 4 
        plt.axvspan(start, end, ymin=0, ymax=0.01, facecolor='red', alpha=0.3)  # alpha控制不透明度
plt.show() 
