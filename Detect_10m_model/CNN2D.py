import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import DataLabel
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
#  not work!
num_classes = 10 
batch_size = 64   
learning_rate = 0.00001
num_epochs = 20  

class MultiLabelDataset(Dataset):
    def __init__(self, img_folder, labels, transform=None):
        self.img_folder = img_folder
        self.img_names = os.listdir(img_folder)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

img_folder = './Fig'
labels = DataLabel.label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# 创建数据集
dataset = MultiLabelDataset(img_folder, labels, transform=transform)

loader = DataLoader(dataset, batch_size=len(dataset))
data = next(iter(loader))
images, _ = data
mean = torch.mean(images, dim=(0, 2, 3))  # 计算均值
std = torch.std(images, dim=(0, 2, 3))    # 计算标准差

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
dataset = MultiLabelDataset(img_folder, labels, transform=transform)

total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size 

# 使用 random_split 进行划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = torch.Generator().manual_seed(42))

# 创建 DataLoader 来进行批处理操作
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = torch.Generator().manual_seed(42))

# 构建模型
model = models.googlenet(weights = None, init_weights = True)  # 不使用预训练权重
# model = models.resnet18(weights = None)
# 替换最后的全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)

if torch.cuda.is_available():
    model = model.cuda()

# 二元交叉熵损失函数，不需要对输出进行sigmoid激活，因为nn.BCEWithLogitsLoss()已经包含了这一步
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.full((1, 10), 4))
if torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 储存路径
work_dir = './CNN'

# 添加tensorboard
writer = SummaryWriter("{}/logs".format(work_dir))

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for i, (inputs, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # 重置梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs).logits
        # outputs = model(inputs)
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
        for imgs, targets in test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # Hamming distance
            predictions = outputs > 0.5
            hamming_distance += (predictions != targets).float().sum(dim=1).mean().item()
 
    print("test set loss: {}".format(total_test_loss/len(test_loader)))
    print("hamming loss: {}".format(hamming_distance/len(test_loader)))
    writer.add_scalar("test_loss", total_test_loss/len(test_loader), epoch)
    writer.add_scalar("hamming_loss", hamming_distance/len(test_loader), epoch)
 
    torch.save(model, "{}/model_{}.pth".format(work_dir,epoch+1))
writer.close()

# model = torch.load('./CNN/model_1.pth')
# model.eval()
# n, m, TP, accuracy = 0, 0, 0, 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         predictions = (outputs > 0.5).float()
#         n += predictions.sum().item()
#         m += labels.sum().item()
#         TP += (predictions*labels).sum().item()
#         accuracy += (predictions == labels).float().sum().item()
# Precision = TP/n
# Recall = TP/m
# F1_score =  2 * Precision * Recall / (Precision + Recall)
# print(f'Accuracy: {accuracy/(10*len(test_dataset))}, Precision: {Precision}, Recall: {Recall}, F1_score: {F1_score}')