import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import DataLabel
import STFTdata
import HFD_data
from PIL import Image
import os
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64   
class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].copy(), dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class CNN1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(input_size, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),                    
            nn.MaxPool1d(3, 2, 1),   
        )
        self.conv2 = nn.Sequential(         
            nn.Conv1d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm1d(128),    
            nn.ReLU(),                  
            nn.MaxPool1d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv1d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm1d(256),    
            nn.ReLU(),                     
            nn.MaxPool1d(2),               
        )
        self.conv4 = nn.Sequential(         
            nn.Conv1d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm1d(512),    
            nn.ReLU(),                     
            nn.MaxPool1d(2),               
        )
        self.conv5 = nn.Sequential(         
            nn.Conv1d(512, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm1d(1024),    
            nn.ReLU(),                                   
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(1024, output_size)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)      
        x = self.out(x)
        return x

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
        
        # 通过全连接层  使用LSTM层的间隔5000/10时间步的输出
        # out = self.fc(out[:, ::500, :])
        # 对于stft_data  使用LSTM层每个时间步的输出
        out = self.fc(out)
        return out.squeeze(-1)

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers    
#         # 输入数据的形状是(batch_size, sequence_length, input_size)   
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)      
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # 初始化隐藏状态和细胞状态
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)       
        
#         # 通过LSTM层 
#         # out的形状为(batch_size, sequence_length, hidden_size)
#         out, _ = self.lstm(x, (h0, c0))
        
#         # 通过全连接层  使用LSTM层的最后一个时间步的输出
#         out = self.fc(out[:, -1, :])
#         return out

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()  

        def block(in_filters, out_filters, bn=True):
            block = [nn.Linear(in_filters, out_filters), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, momentum=0.8))
            return block

        self.model = nn.Sequential(
            *block(input_size, 2048, bn=False),
            *block(2048, 512), 
            *block(512, 128),  
            nn.Linear(128, output_size)                                             
        )

    def forward(self, x):
        out = self.model(x)                       
        return out
    
# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # input and output tensors are provided as (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        transformer_output = self.encoder(src)
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
  
# data = STFTdata.stft_data
# labels = STFTdata.label
data = DataLabel.dataset
labels = DataLabel.label
# data = HFD_data.dataset
# labels = HFD_data.label

# class MultiLabelDataset(Dataset):
#     def __init__(self, img_folder, labels, transform=None):
#         self.img_folder = img_folder
#         self.img_names = os.listdir(img_folder)
#         self.labels = labels
#         self.transform = transform
#     def __len__(self):
#         return len(self.img_names)
#     def __getitem__(self, idx):
#         img_name = self.img_names[idx]
#         img_path = os.path.join(self.img_folder, img_name)
#         image = Image.open(img_path).convert('RGB')
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, torch.tensor(label, dtype=torch.float)
# img_folder = './Fig'
# labels = DataLabel.label
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# # 创建数据集
# dataset = MultiLabelDataset(img_folder, labels, transform=transform)
# loader = DataLoader(dataset, batch_size=len(dataset))
# data = next(iter(loader))
# images, _ = data
# mean = torch.mean(images, dim=(0, 2, 3))  # 计算均值
# std = torch.std(images, dim=(0, 2, 3))    # 计算标准差
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std),
# ])
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataset = MultiLabelDataset(img_folder, labels, transform=transform)
dataset = MultiLabelDataset(data, labels)
total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size 

# 使用 random_split 进行划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = torch.Generator().manual_seed(42))

# 创建 DataLoader 来进行批处理操作
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))

model = torch.load('./CNN1D/model_46.pth')
model.eval()
n, m, TP, accuracy = 0, 0, 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.unsqueeze(1))
        predictions = (outputs > 0.5).float()
        n += predictions.sum().item()
        m += labels.sum().item()
        TP += (predictions*labels).sum().item()
        accuracy += (predictions == labels).float().sum().item()
Precision = TP/n
Recall = TP/m
F1_score =  2 * Precision * Recall / (Precision + Recall)
print(f'Accuracy: {accuracy/(20*len(test_dataset))}, Precision: {Precision}, Recall: {Recall}, F1_score: {F1_score}')