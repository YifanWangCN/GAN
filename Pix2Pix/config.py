import torch
import albumentations as A  # 该库是用来进行数据增强的
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data/maps/train"
VAL_DIR = "data/maps/val"

# 学习率根据论文中的进行的设定
LEARNING_RATE = 2e-4

# 论文里设定的是1 这里变为16， 1的话计算量实在是太大了
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3

# 论文里的Dis的损失函数用到了L1损失函数，该损失函数的权重值设定
L1_LAMBDA = 100
LAMBDA_GP = 10

# 论文里训练了200次
# 从最后的结果来看，500次的效果依然一般 也许可以再训练500次 或者按照论文的要求 batchsize为 1 训练200个epoch
NUM_EPOCHS = 500

# 这里其实是说明是否有预训练过的模型，没有训练过肯定是False了
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

# 对两组图片分别进行resize操作
both_transform = A.Compose(
    [A.Resize(width=256, height=256)]
)

# 对输入图片进行了随机抖动和镜像的操作，并对数据进行了归一化的操作
transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

# 对目标图片只做归一化操作
transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)