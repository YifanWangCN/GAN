import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 首先假设一个自带min batch的张量
# 假设图片的尺寸是16*16
input_img = torch.randn(4, 3, 16, 16).to(DEVICE)


# 接下来构建一个最简单的卷积模型
class Model(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        x = self.model(x)
        return x


model = Model(3,8).to(DEVICE)
output = model(input_img)
print(output.shape)