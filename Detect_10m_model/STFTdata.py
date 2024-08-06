import numpy as np
from scipy import signal
dataset = []
label = []
samples = 100000
stride = 3000
interval = 5000
t = 0
for i in range(6):
    data = np.loadtxt('GZ' + str(i) + '.csv', delimiter=',', unpack=True, encoding='UTF-8')
    point = np.genfromtxt('GZ_label.csv', dtype = float, delimiter=',', usecols=[i], unpack=True, encoding='UTF-8-sig')
    point = point[~np.isnan(point)] 
    for j in range(int((len(data[0])-samples)/stride)+1):
        dataset.append(data[1][stride*j:stride*j+samples])
        label.append([])
        for k in range(int(samples/interval)):
            if any(np.isin(point, data[0][stride*j+k*interval:stride*j+(k+1)*interval]) == True):
                label[t].append(1)
            else:
                label[t].append(0)
        t += 1
    data = data[:, ::-1]
    for j in range(int((len(data[0])-samples)/stride)+1):
        dataset.append(data[1][stride*j:stride*j+samples])
        label.append([])
        for k in range(int(samples/interval)):
            if any(np.isin(point, data[0][stride*j+k*interval:stride*j+(k+1)*interval]) == True):
                label[t].append(1)
            else:
                label[t].append(0)
        t += 1
dataset = np.array(dataset)
stft_data = []
fs = 1000 
for x in dataset:
    f, t, Zxx = signal.stft(x, fs, window='hann', nperseg=512)
    arr = np.array([])
    indexf = np.where((f >= 5) & (f <= 10))[0]
    for k in range(int(samples/interval)):
        indext = np.where((t >= 5*k) & (t < 5*(k+1)))[0]
        arr = np.append(arr, np.mean(np.abs(Zxx[indexf[:, None], indext])**2))
    normalized_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    stft_data.append(normalized_arr)
stft_data = np.array(stft_data)
label = np.array(label)
# print(stft_data[0])
