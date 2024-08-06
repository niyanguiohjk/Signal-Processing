import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import DataLabel
import STFTdata
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].copy(), dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers    
        # 输入数据的形状是(batch_size, sequence_length, input_size)   
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)      
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)       
        
        # 通过LSTM层 
        # out的形状为(batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 通过全连接层  使用LSTM层的最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        return out

data = STFTdata.stft_data
labels = STFTdata.label

# 设置参数
input_size = 1  # 输入特征维度
hidden_size = 64  # 隐藏层维度
output_size = 20  # 输出维度（标签数）
num_layers = 1  # LSTM层的数量
learning_rate = 0.01
batch_size = 64
num_epochs = 40

# 创建数据集和数据加载器
dataset = MultiLabelDataset(data, labels)

total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size 
# 使用 random_split 进行划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = torch.Generator().manual_seed(42))

# 创建 DataLoader 来进行批处理操作
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))

# 实例化模型
model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.full((1, output_size), 4)).to(device)  # 用于多标签分类的损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 储存路径
work_dir = './LSTM'

# 添加tensorboard
writer = SummaryWriter("{}/logs".format(work_dir))

torch.backends.cudnn.enabled = False
# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # 重置梯度
        optimizer.zero_grad()
        
        # 前向传播 二维张量转换为三维张量
        outputs = model(inputs.unsqueeze(-1).contiguous())
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 输出训练信息
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            writer.add_scalar("train_loss", loss.item(), epoch*len(train_loader)+i+1)
    # 更新学习率       
    scheduler.step()
    # 验证模型
    model.eval()  # 设置模型为评估模式
    total_test_loss = 0
    hamming_distance = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(-1).contiguous())
            loss = criterion(outputs, labels)
            total_test_loss = total_test_loss + loss.item()
            # Hamming distance
            predictions = outputs > 0.5
            hamming_distance += (predictions != labels).float().sum(dim=1).mean().item()
 
    print("test set loss: {}".format(total_test_loss/len(test_loader)))
    print("hamming loss: {}".format(hamming_distance/len(test_loader)))
    writer.add_scalar("test_loss", total_test_loss/len(test_loader), epoch)
    writer.add_scalar("hamming_loss", hamming_distance/len(test_loader), epoch)
 
    torch.save(model, "{}/model_{}.pth".format(work_dir,epoch+1))
writer.close()
