import numpy as np
from scipy import signal
def RWJR(filename):
    y = np.loadtxt(filename, delimiter=',', usecols=[1], unpack=True)
    N = len(y)
    b, a = signal.butter(8, 0.008, 'lowpass')
    fy = signal.filtfilt(b, a, y)
    peak1, _ = signal.find_peaks(y, height=0.05, distance=20000)
    valley1, _ = signal.find_peaks(-y, height=0.05, distance=20000)
    peaks = np.zeros(len(peak1), dtype = int)
    valleys = np.zeros(len(valley1), dtype = int)
    u = 0
    for p in peak1:
        k, _ = signal.find_peaks(fy[max(0,p-10000):min(N-1,p+10000)], height=0.01)
        if y[p] >= np.mean(np.sort(fy[k+max(0,p-10000)])[-11:-1]) * 2 and y[p] == np.max(y[max(0,p-10000):min(N-1,p+10000)]):
            peaks[u] = p
            u += 1
    w = 0
    for v in valley1:
        l, _ = signal.find_peaks(-fy[max(0,v-10000):min(N-1,v+10000)], height=0.01)
        if y[v] <= np.mean(np.sort(fy[l+max(0,v-10000)])[1:11]) * 2 and y[v] == np.min(y[max(0,v-10000):min(N-1,v+10000)]):
            valleys[w] = v
            w += 1
    g = np.sort(np.append(np.trim_zeros(peaks), np.trim_zeros(valleys)))
    m = len(g)
    i = 0
    j = 0
    t = np.zeros(m, dtype = int) #判断当前峰值是否属于接头，值1是接头，1个接头只算1个波峰或波谷
    t[0] = 1
    h = np.zeros((2,m), dtype = int) #第二行值0为低接头，值2为有缝接头，值1为凸接头
    while i < m-1:
        if g[i+1] - g[i] < 5000:
            q = np.argmax(np.array([y[g[i]] + 0.1 if y[g[i]] > 0 else abs(y[g[i]]), y[g[i+1]] + 0.1 if y[g[i+1]] > 0 else abs(y[g[i+1]])]))
            h[0][j] = g[q+i]
            if y[h[0][j]] > 0:
                h[1][j] = 1
            if y[h[0][j]] < 0:
                if y[h[0][j]-10] - y[h[0][j]] >= 0.2 and y[h[0][j]+10] - y[h[0][j]] >= 0.2:
                    h[1][j] = 2
                else:
                    h[1][j] = 0
            t[q+i] = 1
            t[1-q+i] = 0
            j = j + 1
        if 5000 <= g[i+1] - g[i] < 20000:
            if t[i] == 0:
                t[i+1] = 1
            if t[i] == 1 and g[i] not in h[0]:
                if abs(y[g[i]]) >= abs(y[g[i+1]]) or g[i] - h[0][j-1] >= 25000:
                    h[0][j] = g[i]
                    if y[h[0][j]] <= -0.1 - np.max(y[max(0,g[i]-5000):min(N-1,g[i]+5000)]):
                        if y[h[0][j]-10] - y[h[0][j]] >= 0.2 and y[h[0][j]+10] - y[h[0][j]] >= 0.2:
                            h[1][j] = 2
                        else:
                            h[1][j] = 0
                    else:
                        h[1][j] = 1
                    j = j + 1
                else:
                    t[i] = 0
                    t[i+1] = 1
        if g[i+1] - g[i] >= 20000:
            t[i+1] = 1
            if t[i] == 1 and g[i] not in h[0]:
                h[0][j] = g[i]
                if y[h[0][j]] <= -0.1 - np.max(y[max(0,g[i]-5000):min(N-1,g[i]+5000)]):
                    if y[h[0][j]-10] - y[h[0][j]] >= 0.2 and y[h[0][j]+10] - y[h[0][j]] >= 0.2:
                        h[1][j] = 2
                    else:
                        h[1][j] = 0
                else:
                    h[1][j] = 1
                j = j + 1
        i = i + 1
    if t[i] == 1 and g[i] not in h[0]:
        h[0][j] = g[i]
        if y[h[0][j]] <= -0.1 - np.max(y[max(0,g[i]-5000):min(N-1,g[i]+5000)]):
            if y[h[0][j]-10] - y[h[0][j]] >= 0.2 and y[h[0][j]+10] - y[h[0][j]] >= 0.2:
                h[1][j] = 2
            else:
                h[1][j] = 0
        else:
            h[1][j] = 1
    s = h[:, :np.count_nonzero(h[0])]
    RWJ = y[np.clip(np.array([np.arange(-2500, 2501) + c for c in s[0]]), 0, N-1)]
    return RWJ, s[1]
