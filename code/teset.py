import random
import torch
import torch.nn as nn
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import newdata as D
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset

# 加载模型
num_classes = 61  # 根据数据集的类别数量设置
model = resnet50(weights=None)  # 不加载预训练权重
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(torch.load('D:/homework/shixi/plant/model/plant.pth'))
model.eval()  # 切换模型到评估模式


# 加载测试数据
test_data = D.PlantDataSet(mode='test')

# 设置显示中文，指定一个支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题

# 数据的行数列数
row, col = (5, 5)
fig, axs = plt.subplots(row, col, constrained_layout=True, figsize=(10, 10))
fig.suptitle('植物表型特征识别', fontsize=30)

for i in range(row):
    for j in range(col):
        data = random.choice(test_data)
        img = data[0]
        real_label = test_data.label_name(data[1])

        # 模型预测
        img_tensor = img.unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            result = model(img_tensor)
            result = torch.nn.functional.softmax(result, dim=1)

        # 找到可能性最高的元素的索引，并将其转换为对应的中文标签
        label = test_data.label_name(torch.argmax(result).item())
        # 将概率保留2位小数
        prob = str(torch.max(result).item())[:4]

        # 预测结果和真实的结果
        if label != real_label:
            axs[i][j].set_title(f'预测结果:{label}\n概率:{prob}\n正确结果:{real_label}', fontdict={'color': 'red'})
        else:
            axs[i][j].set_title(f'预测结果:{label}\n概率:{prob}\n正确结果:{real_label}')

        # 展示测试图像
        img_np = img.permute(1, 2, 0).numpy()  # 将(c, h, w)调整为(h, w, c)
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 反归一化
        img_np = img_np.clip(0, 1)  # 确保值在[0, 1]范围内
        axs[i][j].imshow(img_np)

plt.show()
