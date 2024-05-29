import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import shap

# 读取Excel文件的第二个工作表
file_path = 'output.xlsx'
new_df = pd.read_excel(file_path)

# 超参数定义
epochs = 100
batch_size = 32
learning_rate = 0.005
dropout_rate = 0.5

# 数据预处理
# 将 DataFrame 转换为 numpy 数组
X = new_df.drop(columns=['ans']).values
y = new_df['ans'].values

# 处理标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 定义全连接神经网络模型
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(label_encoder.classes_))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FullyConnectedNN()

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 使用 SHAP 解释模型
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 打印解释结果
shap.summary_plot(shap_values, X_test, feature_names=new_df.drop(columns=['ans']).columns)
