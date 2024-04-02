import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm


# 读取数据并且分割输入和标签
def read_data(train_or_test, num=0):
    with open(os.path.join("data", train_or_test + '.txt'), encoding='utf-8') as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        t, l = data.split("\t")
        texts.append(t)
        labels.append(l)

    if num == 0:
        return texts, labels
    else:
        return texts[0:num], labels[0:num]


# 构建语料库
def built_corpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # pad用于填充词向量，保证每个词语的维度一致，unk用于补充没有出现在语料库的词语
    for text in train_texts:
        for word in text:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)  # 大小为len(word_2_index)*embedding_num)的随机矩阵


# 构建数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_2_index, max_len):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        # 1. 根据index获取数据
        text = self.texts[index]  # 注意这里text表示一个新闻标题，而一个新闻标题有多个字符串，故text也是一个列表
        label = int(self.labels[index])
        # 2. 填充,裁剪数据长度至max_len
        text = text[:self.max_len]  # 具体来说是将超过max_len长度的句子裁剪，不满的保留
        # 3. 将文本转为向量
        text_index = [self.word_2_index.get(i, 1) for i in text]  # 将文本中的每个单词转换为其对应的索引值（如果单词不在word_2_index中，则使用默认值1）。
        text_index = text_index + [0] * (self.max_len - len(text_index))  # 句子长度不满max_len的补0

        text_index = torch.tensor(text_index).unsqueeze(dim=0)  # 将索引列表转换为PyTorch张量，并使用unsqueeze方法增加一个维度

        return text_index, label

    def __len__(self):
        return len(self.labels)


# 训练每一类卷积核
class Block(nn.Module):
    def __init__(self, kernel_s, embedding_num, max_len, hidden_num):
        super().__init__()
        # 每句话的总大小为(batch_size*in_channel*max_len*embedding_num)，其中1为添加的额外通道，只是为了计算，因为Conv2d要求4维文本向量
        # 对于文本而言，卷积通道为一，定义输出通道为hidden_num，即每一类卷积核的数量，卷积核的行数kernel_s为超参数，列数即每个词的向量长度
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embedding_num))
        self.act = nn.ReLU()
        self.maxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))  # kernel_size准确来说应该为卷积完成后的size

    # 传入词向量开始卷积
    def forward(self, batch_emb):
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)  # 移除最后一个维度，以便与后面的池化层兼容
        m = self.maxp.forward(a)
        m = m.squeeze(dim=-1)  # 应用最大池化操作，并再次移除最后一个维度，与全连接层兼容
        return m


# 构建模型
class TextCnn(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]   # 128
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 定义一个线性分类器，输入是三个Block块的共hidden_num * 3个输出的拼接，输出是类别数。
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, batch_label=None):
        batch_emb = self.emb_matrix(x)   # 特殊索引，用于将输入的文本数据 x 转换为词嵌入矩阵
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)

        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)  # 将所有向量拼接
        pre = self.classifier(feature)  # 拼接向量进入全连接层分类

        # 不填batch_label是为了测试
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)   # 输出最大值所代表的类别


if __name__ == "__main__":
    train_texts, train_labels = read_data("train")
    test_texts, test_labels = read_data("test")

    # 超参数
    batch_size = 64
    max_len = 30
    embedding_num = 128
    epochs = 10
    class_num = len(set(train_labels))
    lr = 0.01
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hidden_num = 10

    # 得到每个词的字典和词
    word_2_index, words_embedding = built_corpus(train_texts, embedding_num)

    # 带入数据，将文本转化为数据矩阵
    train_dataset = TextDataset(train_texts, train_labels, word_2_index, max_len)
    # 数据加载，会先经过dataset,加载过程中根据batch_size取对应条数据和标签，然后挨个经过__getitem__函数处理，最后整合return
    # 整个train_dataloader包含两个张量，一个是大小为 batch_size*max_len*embedding_num的数据集张量，一个是大小为batch_size的标签张量
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)  # 在dataloader中数值型都会转化为tensor

    test_dataset = TextDataset(test_texts, test_labels, word_2_index, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = TextCnn(words_embedding, max_len, class_num, hidden_num)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(model.emb_num)
    # 训练模型
    for epoch in range(epochs):
        for x, batch_label in tqdm(train_dataloader):
            x, batch_label = x.to(device), batch_label.to(device)
            loss = model.forward(x, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {loss:.3f}")

    # 测试模型
    model.eval()  # 关闭dropout
    with torch.no_grad():  # 关闭梯度计算
        correct = 0
        for x, batch_label in test_dataloader:
            x, batch_label = x.to(device), batch_label.to(device)
            pre = model.forward(x)
            correct += int(torch.sum(pre == batch_label))

    print(f"acc={correct / len(test_texts):.2f}")
