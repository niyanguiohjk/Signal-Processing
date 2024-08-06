import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import DataLabel
import STFTdata
import math
# 不合适此任务 需要对decoder的输入输出数据添加起始符和终止符
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].copy(), dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
    
# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dim_feedforward=2048,
                                          dropout=0.1, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)) # 序列长度

        transformer_output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(transformer_output)
        return output.squeeze(-1)
    
# 位置编码器不改变size
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

data = DataLabel.dataset
labels = DataLabel.label

# 参数设置
input_dim = 1  # 输入数据的特征维度
output_dim = 1 
d_model = 512
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
model = TransformerModel(input_dim, d_model, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.full((1, 10), 4)).to(device)  # 用于多标签分类的损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 储存路径
work_dir = './Transformer'

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
        outputs = model(inputs.unsqueeze(-1), labels.unsqueeze(-1))
        
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
            outputs = model(inputs.unsqueeze(-1), labels.unsqueeze(-1))
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
