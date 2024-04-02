import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

# 超参数设置
batch_size = 32
lr = 0.01
epochs = 10
device = "cuda:0" if torch.cuda.is_available() else "cpu"

data_train = pd.read_csv('data/fashion-mnist_train.csv')
data_test = pd.read_csv('data/fashion-mnist_test.csv')

labels_train = data_train.iloc[:, 0].values
pixels_train = data_train.iloc[:, 1:].values
labels_test = data_test.iloc[:, 0].values
pixels_test = data_test.iloc[:, 1:].values


# 进行特征放缩，提高学习效率
pixels_train = torch.tensor(pixels_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0  # -1表示不知道这一维度的大小，会根据其他维度大小自动计算
pixels_test = torch.tensor(pixels_test, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0

labels_train = torch.tensor(labels_train, dtype=torch.long).view(-1)  # 转换为 PyTorch 张量，并扁平化为一维张量
labels_test = torch.tensor(labels_test, dtype=torch.long).view(-1)

# 接受的两个参数的第一个维度必需相等
train_dataset = TensorDataset(pixels_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = TensorDataset(pixels_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化核大小为2x2，步长为2，挨个扫描特征图将特征图缩小，可以计算得到特征图的长宽均缩小为原来的一半

        # 全连接层
        self.fc1 = nn.Linear(32 * 14 * 14, 1024)  # 32是输出通道，即卷积核个数，14*14是经过池化后的特征图大小
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        c = self.conv2d(x)
        a = self.act1(c)
        m = self.maxp(a)

        x = m.view(m.size(0), -1)  # 扁平化特征图，进入全连接层分类
        x = self.fc1(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if y is not None:
            loss = self.loss_fun(x, y)
            return loss
        else:
            return torch.argmax(x, dim=-1)


model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(epochs):
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        loss = model.forward(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {loss:.3f}")

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pre = model.forward(x)
        correct += int(torch.sum(pre == y))

print(f"acc={correct / len(test_dataset):.2f}")

