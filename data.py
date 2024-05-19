import os
import torch
import torchvision.transforms as T
from PIL import Image


class PlantDataSet(torch.utils.data.Dataset):
    data_path = "D:\桌面D\plant\PlantData\dataset"
    label_path = "D:\桌面D\plant\PlantData\labels.txt"

    def __init__(self, mode='train'):
        super(PlantDataSet, self).__init__()
        # 数据集模式
        self.mode = mode.lower()
        assert mode.lower() in ['train', 'val', 'test'], \
            "mode应为'train'、'val'或'test'，但传入的是{}".format(mode)
        # 加载标签文件
        with open(self.label_path, 'r', encoding='utf-8') as labels:
            self.label_names = [label.removesuffix('\n') for label in labels.readlines()]
        size = (224, 224)
        if self.mode != 'test':
            # 数据集图像增强
            self.transform = T.Compose([
                T.Resize(size=size),  # 统一图片大小
                T.RandomHorizontalFlip(),  # 随机水平翻转图片
                T.RandomRotation((0, 360)),  # 随机旋转图片
                T.RandomVerticalFlip(),  # 随机垂直翻转图片
                T.ToTensor(),  # 转换为Tensor
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # 归一化
            ])
        else:
            # 测试集进行图片预处理
            self.transform = T.Compose([
                T.Resize(size=size),  # 统一图片大小
                T.ToTensor(),  # 转换为Tensor
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # 归一化
            ])
        # 准备数据集
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for label in self.label_names:
            image_dir = os.path.join(self.data_path, self.mode, label)
            for img in os.listdir(image_dir):
                img = Image.open(os.path.join(image_dir, img))
                img = img.convert("RGB")
                data.append((img, self.label_names.index(label)))
        return data

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
