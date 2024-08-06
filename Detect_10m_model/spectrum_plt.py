# data = np.loadtxt('D1_L0.csv', delimiter=',', unpack=True, encoding='UTF-8')
# point = np.genfromtxt('D_label.csv', dtype = float, delimiter=',', usecols=[0], unpack=True, encoding='UTF-8-sig')
# point = point[~np.isnan(point)].astype(int)  
# plt.title('RWJ')
# plt.xlabel("mileage/mm")
# plt.ylabel("amplitude/mm")
# plt.plot(range(1, len(data)+1), data)
# plt.plot(point, data[point-1], ".")
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import DataLabel
# import shutil
# import os
fs = 1000 
for i, x in enumerate(DataLabel.dataset):
    f, t, Zxx = signal.stft(x, fs, window='hann', nperseg=512) 
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.01, shading='gouraud')
    plt.ylim(5, 10)
    plt.axis('off')  # 关闭坐标轴显示
    # 保存图像
    plt.savefig('./Fig/'+str(i)+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像，以便释放资源
# for j in range(Dataset.t):
#     for k in range(10):
#         if Dataset.label[j][k] == 1:
#             shutil.copy('./Fig/'+str(j)+'.png', './Fig/'+str(k))
#     os.remove('./Fig/'+str(j)+'.png')