import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision import transforms


class PlantDataSet(Dataset):
    data_path = 'D:/homework/shixi/plant/PlantData/dataset'
    label_path = 'D:/homework/shixi/plant/PlantData/labels.txt'

    def __init__(self, mode='train'):
        super(PlantDataSet, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['train', 'val', 'test'], \
            f"mode应为'train'、'val'或'test'，但传入的是{mode}"

        with open(self.label_path, 'r', encoding='utf-8') as labels:
            self.label_names = [label.strip() for label in labels.readlines()]

        # Debug: 打印标签
        print(f"Labels: {self.label_names}")
        print(f"Number of labels: {len(self.label_names)}")

        size = (224, 224)
        if self.mode != 'test':
            self.transform = transforms.Compose([
                Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 360)),
                transforms.RandomVerticalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                Resize(size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for label in self.label_names:
            image_dir = os.path.join(self.data_path, self.mode, label)
            if not os.path.isdir(image_dir):
                print(f"Warning: Directory {image_dir} does not exist.")
                continue
            for img_file in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_file)
                data.append((img_path, self.label_names.index(label)))
        return data

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    def label_name(self, label):
        return self.label_names[label]
