import numpy as np
dataset = []
label = []
samples = 50000
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
dataset = np.mean(dataset.reshape(dataset.shape[0], -1, 10), axis=2) # 作RNN时使用 防止序列过长导致的内存溢出和训练缓慢
dataset = 2 * (dataset - np.min(dataset, axis=1, keepdims=True)) / (np.max(dataset, axis=1, keepdims=True) - np.min(dataset, axis=1, keepdims=True)) - 1 # 归一化到[-1,1]