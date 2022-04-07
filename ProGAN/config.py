import cv2
import torch
from math import log2

# 一开始训练的是 4*4的图片
START_TRAIN_AT_IMG_SIZE = 4

DATASET = "celeba_hq"
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_MODEL = True
LOAD_MODEL = False

LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3

Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper

CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)

# 这个主要是用来看训练成果的
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 2

#  print(PROGRESSIVE_EPOCHS)