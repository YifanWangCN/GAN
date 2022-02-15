from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

# 数据集的下载连接为 https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

class RealVanDataset(Dataset):
    def __init__(self, root_real, root_van, transform=None):

        self.root_real = root_real
        self.root_van = root_van
        self.transform = transform

        self.real_images = os.listdir(root_real)  # 真实世界的图片
        self.van_images = os.listdir(root_van)  # 梵高的画作图片

        # check两个集里面照片最多的那个
        # van有 755 张图片 real有1000张图片
        self.length_dataset = max(len(self.real_images), len(self.van_images))

        self.real_len = len(self.real_images)
        self.van_len = len(self.van_images)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):

        # 取余的方式是保证这个索引不会超出范围，因为两个数据其实长度是不一样的
        real_img = self.real_images[index % self.real_len]
        van_img = self.van_images[index % self.van_len]

        real_path = os.path.join(self.root_real, real_img)
        van_path = os.path.join(self.root_van, van_img)

        # 将图片转为数组形式，为了后续的数据增强
        real_img = np.array(Image.open(real_path).convert("RGB"))
        van_img = np.array(Image.open(van_path).convert("RGB"))

        # 接下来是进行数据增强
        if self.transform:
            augmentations = self.transform(image=real_img, image0=van_img)
            real_img = augmentations["image"]
            van_img = augmentations["image0"]

        return real_img, van_img

