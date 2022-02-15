import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 该文件保存本次任务主要的一些参数设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

BATCH_SIZE = 1

LEARNING_RATE = 1e-5

LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

NUM_WORKERS = 4
NUM_EPOCHS = 100

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_Van = "genvan.pth.tar"
CHECKPOINT_GEN_Real = "genreal.pth.tar"
CHECKPOINT_CRITIC_Van = "criticvan.pth.tar"
CHECKPOINT_CRITIC_Real = "critiReal.pth.tar"

# 对图片进行裁剪，镜像，归一化的操作
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)