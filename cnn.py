import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split


#  定义device，将后续数据搬到gpu上运行
device = torch.device("cuda")

# 加载数据读取数据，并且进行预处理
data = pd.read_csv("data-without bmi.csv")

# 定义字典，将类别映射到数值
obesityCategory_to_number = {
    'Underweight': 0,
    'Normal weight': 1,
    'Overweight': 2,
    'Obese': 3
}
gender_to_number = {
    'Male': 1,
    'Female': 0
}
# 使用.replace()函数进行替换
data['ObesityCategory'] = data['ObesityCategory'].replace(obesityCategory_to_number)
data['Gender'] = data['Gender'].replace(gender_to_number)

X = data.iloc[:, :-1].values  # 取前5列作为特征值。并将转化为numpy数组[:, :-1]前一个：选取行，后一个：选取列
y = data.iloc[:, -1].values  # 取后一列作为标签

y = y.astype(int)   # numpy数组的用法，强制类型转换，多分类问题标签为整数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # 分割数据

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器，TensorDataset接收特征和标签作为输入，并返回一个可以迭代的数据集对象。DataLoader将数据集分成小批量。 在每个epoch中打乱数据
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # in_channels  词向量维度，处理该问题的时候输入特征都是一维的标量，词向量也只能是一维
        # out_channels 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
        # kernel_size  卷积核的大小,第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        # stride       卷积步长，卷积核每次扫描后移动的距离
        # padding      在输入数据的两侧填充0，以保持空间维度不变
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()    # 采用relu激活函数
        self.fc1 = nn.Linear(16 * 5, 64)   # 第一个全连接层，16*5表示将16个卷积核和五个特征值神经元连接64表示该层的神经元个数
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 4)  # 输出层，4表示输出4类结果

    # 前向传播
    def forward(self, x):
        # 添加一个额外的通道维度，因为卷积核需要扫描二维平面，将每个输入样本转换为一个单通道的图像（宽度为5），然后传递给卷积层
        # -1表示根据张量的总元素数量和其它已指定维度的大小自动计算该维度
        x = x.view(-1, 1, 5)
        x = self.conv1(x)  # 特征值通过卷积核
        x = self.relu(x)   # 卷积后进行非特征激活
        # 修改张量维度适应全连接层，x.size(0)表示第一维大小，也就是bath_size,
        x = x.view(x.size(0), -1)
        # 进入全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 初始化模型并且选择损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model = model.to(device)  # 将模型移动到gpu
# 训练模型
epochs = 10  # 定义整个数据集被训练的次数
for epoch in range(epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 转移数据到gpu
        # 计算损失函数，train_loader是一个已经包含了 X_train_tensor, y_train_tensor的可迭代对象
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()   # 梯度清零，因为每次计算都会保留上一次梯度
        loss.backward()         # 计算获取当前梯度
        optimizer.step()        # 根据梯度更新模型参数

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')  # 每选练完一次数据集时打印loss的值，方便了解训练过程

# 测试模型，会自动关闭dropout
model.eval()
with torch.no_grad():   # 关闭梯度计算，测试不需要计算梯度
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probability, predicted = torch.max(outputs, 1)   # 返回输出张量(为各个类别的)的概率值和对于的类别，因为输出结果默认表示概率,
        total += targets.size(0)               # targets.size(0)返回targets(y_test_tensor)的第一维大小，也就是行数表示测试集合的样本个数
        correct += (predicted == targets).sum().item()  # 返回

accuracy = correct / total
print(f'accuracy = {accuracy}')