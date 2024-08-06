import numpy as np
a1 = np.loadtxt('RWJGZ1_L.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a2 = np.loadtxt('RWJGZ1_R.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a3 = np.loadtxt('RWJGZ2_L.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a4 = np.loadtxt('RWJGZ2_R.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a5 = np.loadtxt('RWJGZ3_L.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a6 = np.loadtxt('RWJGZ3_R.csv', delimiter=',', usecols=[0], unpack=True, encoding='UTF-8')
a = np.concatenate((a1, a2, a3, a4, a5, a6))
b1 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[0], unpack=True, encoding='UTF-8-sig')
b2 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[1], unpack=True, encoding='UTF-8-sig')
b3 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[2], unpack=True, encoding='UTF-8-sig')
b4 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[3], unpack=True, encoding='UTF-8-sig')
b5 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[4], unpack=True, encoding='UTF-8-sig')
b6 = np.genfromtxt('RWJGZ_ST.csv', dtype = float, delimiter=',', usecols=[5], unpack=True, encoding='UTF-8-sig')
b = np.concatenate((b1, b2, b3, b4, b5, b6))
b = b[~np.isnan(b)]
count = 0
for i in range(len(a)):
    for j in range(len(b)):
        if abs(a[i] - b[j]) <= 1:
            count += 1
            break
print(count)
print(len(a))
print(len(b))