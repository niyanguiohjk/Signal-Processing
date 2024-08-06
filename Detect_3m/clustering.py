from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt 
from RWJR import RWJR

waveform1, lable1 = RWJR('dataK034717_1.csv')
waveform2, lable2 = RWJR('dataK034717_2.csv')
waveform3, lable3 = RWJR('dataK035730_1.csv')
waveform4, lable4 = RWJR('dataK035730_2.csv')
waveform5, lable5 = RWJR('dataK041900_1.csv')
waveform6, lable6 = RWJR('dataK041900_2.csv')
waveform = np.concatenate((waveform1, waveform2, waveform3, waveform4, waveform5, waveform6), axis=0)
lable = np.concatenate((lable1, lable2, lable3, lable4, lable5, lable6), axis=0)
print(waveform.shape, lable.shape)

f1 = np.array([np.mean(x) for x in waveform])
f2 = np.array([np.var(x) for x in waveform])
f3 = np.array([np.max(x) for x in waveform])
f4 = np.array([np.min(x) for x in waveform])
f5 = f3 + f4 + 0.1
f6 = f3 - np.array([np.mean(np.sort(x)[-1600:-100]) for x in waveform])
f7 = f4 - np.array([np.mean(np.sort(x)[100:1600]) for x in waveform])
f8 = f6 + f7
f9 = np.array([x[np.argmin(x)-10]+x[np.argmin(x)+10]-np.min(x)*2 for x in waveform]) - 0.4
# print(np.sort(f5)[:10])
feature = np.column_stack((f1, f2, f5, f8, f9))
print(feature.shape)

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(feature)

plt.figure()
plt.scatter(X_tsne[lable == 0, 0], X_tsne[lable == 0, 1], c='r', label='RWJ0')
plt.scatter(X_tsne[lable == 1, 0], X_tsne[lable == 1, 1], c='b', label='RWJ1')
plt.scatter(X_tsne[lable == 2, 0], X_tsne[lable == 2, 1], c='y', label='RWJ2')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Visualization of RWJs')
plt.grid()
plt.legend()
plt.show()
