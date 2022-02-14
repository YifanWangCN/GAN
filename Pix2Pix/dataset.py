from PIL import Image   # 该库可以用来从文件夹中读取图片
import numpy as np
import os
import config
from torch.utils.data import Dataset    # 该库是可以用来讲图片进行分配batch的

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)  # 遍历文件夹内的图片

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):

        # 获取文件名
        img_file =self.list_files[index]

        # 每一张图片的地址
        img_path = os.path.join(self.root_dir, img_file)

        # 将地址图片打开，将数据转换为数组形式
        image = np.array(Image.open(img_path))

        # 将原图像的两个部分进行分割
        input_image = image[ : , :600, : ]
        target_image = image[ : ,600: , : ]

        # 对两个图像进行数据增强处理
        # 首先将两个图像进行resize到256*256的尺寸
        augmentations1 = config.both_transform(image=input_image)
        augmentations2 = config.both_transform(image=target_image)
        input_image = augmentations1["image"]
        target_image = augmentations2["image"]

        # 对input_image进行了归一化，随机镜像和抖动的操作，论文里面是有这个操作的
        input_image = config.transform_only_input(image=input_image)["image"]

        # 对目标图片进行归一化操作，这也是为了在判别器里进行操作
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image,target_image

