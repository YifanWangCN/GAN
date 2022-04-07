import torch


# 下面是模型超参数的一些设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 完全按照论文中的参数进行了设计
LEARNING_RATE = 2e-4
BATCH_SIZE = 128

Data_root = "celeb_dataset"

# 输入的随机数
Z_DIM = 100

# 训练次数
NUM_EPOCH = 10

NUM_WORKER = 2

