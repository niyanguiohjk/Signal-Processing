import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
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
dataset = dataset.reshape(-1, 5000)
dataset = 2 * (dataset - np.min(dataset, axis=1, keepdims=True)) / (np.max(dataset, axis=1, keepdims=True) - np.min(dataset, axis=1, keepdims=True)) - 1 # 归一化到[-1,1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiLabelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].copy(), dtype=torch.float32)

## ##### 定义判别器 Discriminator ######
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),                                                           
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
        validity = self.model(x)                        ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率
    
## ###### 定义生成器 Generator #####
## 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
## 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
## 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):           ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]             ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))    ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))      ## 非线性激活函数
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),     
            *block(128, 256),                                   ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),                                   ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),                                  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            *block(1024, 2048),
            *block(2048, 4096), 
            nn.Linear(4096, input_dim),                         
            nn.Tanh()                                           ## 将数据每一个都映射到[-1, 1]之间
        )
    ## view():相当于numpy中的reshape，重新定义矩阵的形状
    def forward(self, z):                                       ## 输入的是(64， 100)的噪声数据
        x = self.model(z)                                       ## 噪声数据通过生成器模型
        return x                     
                             
data = dataset

# 参数设置
input_dim = 5000
latent_dim = 100
batch_size = 64
num_epochs = 30

# 创建数据集和数据加载器
dataset = MultiLabelDataset(data)

total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size 
# 使用 random_split 进行划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = torch.Generator().manual_seed(42))

# 创建 DataLoader 来进行批处理操作
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))

# 实例化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss().to(device)  # 用于二分类的损失函数

optimizer_G = torch.optim.AdamW(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 学习率调度器
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs)

work_dir = './GAN'
writer = SummaryWriter("{}/logs".format(work_dir))
## ----------
##  Training
## ----------
## 进行多个epoch的训练
for epoch in range(num_epochs):                               
    for i, inputs in enumerate(train_loader):                   
        ## =============================训练判别器==================
        inputs = inputs.to(device)
        real_label = torch.ones(inputs.size(0), 1).to(device)      ## 定义真实的label为1
        fake_label = torch.zeros(inputs.size(0), 1).to(device)     ## 定义假的label为0
        ## ---------------------
        ##  Train Discriminator
        ## 分为两部分：1、真的判别为真；2、假的判别为假
        ## ---------------------
        ## 计算真实信号的损失
        real_out = discriminator(inputs)                         
        loss_real_D = criterion(real_out, real_label)              
        real_scores = real_out                                     
        ## 计算假信号的损失
        ## detach(): 从当前计算图中分离下来避免梯度传到G，因为G不用更新
        z = torch.randn(inputs.size(0), latent_dim).to(device)     ## 随机生成一些噪声
        fake_signal = generator(z).detach()                           
        fake_out = discriminator(fake_signal)                                
        loss_fake_D = criterion(fake_out, fake_label)                     
        fake_scores = fake_out                                            
        ## 损失函数和优化
        loss_D = loss_real_D + loss_fake_D                         ## 损失包括判真损失和判假损失
        optimizer_D.zero_grad()                            
        loss_D.backward()                                   
        optimizer_D.step()                                  

        ## -----------------
        ##  Train Generator
        ## -----------------
        z = torch.randn(inputs.size(0), latent_dim).to(device)               
        fake_signal = generator(z)                                            
        output = discriminator(fake_signal)                                    
        ## 损失函数和优化
        loss_G = criterion(output, real_label)                             
        optimizer_G.zero_grad()                                             
        loss_G.backward()                                                   
        optimizer_G.step()                                                 

        ## 打印训练过程中的日志
        ## item():取出单元素张量的元素值并返回该值，保持原元素类型不变
        if (i + 1) % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D real: %f] [D fake: %f]"
                % (epoch+1, num_epochs, i+1, len(train_loader), loss_D.item(), loss_G.item(), real_scores.data.mean(), fake_scores.data.mean())
            )
            writer.add_scalars("losses", {"loss_D": loss_D.item(), "loss_G": loss_G.item()}, epoch*len(train_loader)+i+1)
    scheduler_G.step()
    scheduler_D.step()
writer.close()
## 保存模型
torch.save(generator, './GAN/generator.pth')
torch.save(discriminator, './GAN/discriminator.pth')
