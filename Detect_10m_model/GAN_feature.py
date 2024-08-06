import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
dataset = dataset.reshape(-1, 5000)
dataset = 2 * (dataset - np.min(dataset, axis=1, keepdims=True)) / (np.max(dataset, axis=1, keepdims=True) - np.min(dataset, axis=1, keepdims=True)) - 1 # 归一化到[-1,1]
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5000, 2048),                                                           
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),                        
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),  
            nn.Linear(1024, 512),                        
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.2), 
            # nn.Linear(2048, 512),                        
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),  
            nn.Linear(512, 256),                        
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Dropout(0.2),       
            nn.Linear(256, 1)                                                      
        )

    def forward(self, x):
        validity = self.model(x)                        
        return validity                                          
                          
model = torch.load('./GAN/discriminator.pth')
model.eval()   
model = nn.Sequential(*list(model.model.children())[:-2])
Feature = model(torch.tensor(dataset, dtype=torch.float32).to(device))
Feature = Feature.view(-1,10,Feature.shape[-1])
# print(Feature.shape)