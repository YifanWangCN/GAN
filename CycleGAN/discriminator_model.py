import torch
import torch.nn as nn
from torchsummary import summary # 做测试用的

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)


# 该判别器依然用的是 70*70 PatchGAN 即全卷积神经网络
# 论文中  The discriminator architecture is: C64-C128-C256-C512
class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64,128,256,512]):
        super().__init__()

        # 论文中 We do not use InstanceNorm for the first C64 layer.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        # 这个sigmoid的设计在pix2pix中被使用
        return torch.sigmoid(self.model(x))


# 用来测试所用
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = Discriminator(in_channels=3).to(DEVICE)
# summary(model, input_size=(3, 256, 256))

