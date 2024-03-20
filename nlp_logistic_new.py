import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 读取数据集
data = pd.read_csv('data.csv')
# 创建标签编码器
le = LabelEncoder()
# fit_transform函数进行标签值转化，第一个标签性别为0，第二个为1，以此类推
data['Gender'] = le.fit_transform(data['Gender'])


# 自定义映射函数，将PhysicalActivityLevel列结果分为两类，不高于2级记1，反之记作0
def custom_map1(value):
    if value <= 2:
        return 0
    else:
        return 1


# 计算新的特征值
# data的map函数接受一个函数，将data的数据作为可迭代序列，并将其作用于参数函数
data['PhysicalActivityLevel'] = data['PhysicalActivityLevel'].map(custom_map1)
# 删除不需要的列，axis=1表示列，0表示行
X = data[['Weight', 'Age', 'Height', 'Gender']]
Y = data['PhysicalActivityLevel']

# 划分训练集和测试集,其中训练集占80%，测试集占20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建逻辑回归模型并训练,调整参数，c表示学习率与惩罚项的参数比的倒数，c越大模型越容易过拟合
model = LogisticRegression(penalty='l2', tol=1e-5, solver='sag', max_iter=10000, C=0.01)
model.fit(X_train, Y_train)

# 预测测试集结果
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)


# 将新的数据存储在字典中
new_data = {
    'Weight': 60,
    'Age': 30,
    'Height': 140,
    'Gender': 1,
}


# 将新数据转换为和训练集一样的格式
new_data = pd.DataFrame([new_data])

# 使用训练好的模型进行预测
prediction = model.predict(new_data)
pre_label = ['Obese' if pre == 0 else 'slender' for pre in prediction]

# 输出预测结果
print("Predicted PhysicalActivityLevel:", pre_label[0])