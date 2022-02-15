import torch
import torch.nn as nn
from torchsummary import summary # 做测试用的


# 需要申明的是本次复现代码主要是对 256*256的条件图片做的
class ConvBlock(nn.Module):

    # **kwargs 表示不确定传入的参数的长度，该参数必须为字典形式，即 name = “Steven”
    # 这个地方的**kwargs 是用来包含 卷积核尺寸，步长，填充这三个参数的
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()

        # 论文中Reflection padding was used to reduce artifacts
        # nn.Identity()模块是不改变输入直接返回，应该是为后续残差网络做准备的
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )


    def forward(self,x):
        return self.conv(x)


# 深度残差网络的BLOCK
# 深度残差网络是由两个相同节点的层构建的，并包含一个skip connection
# 残差网络下默认3*3卷积下padding为1，stride也为1，保证尺寸不会改变
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(channels,channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self,x):
        # 残差网络精妙的地方，注意这个返回值是还没有经过激活函数的值
        return x + self.block(x)



# 接下里就是编写生成器的模型了
# 因为我们训练的图片是256*256的图片，所以按照论文里面的设定，需要9个残差网络
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
class Generator(nn.Module):
    def __init__(self, img_channels,num_features=64, num_residuals=9):
        super().__init__()

        #  首先是一个c7s1-64, 表示输出特征图为64，步长为1，kernel_size=7,源码中padding设的值为3
        #  为什么第一个不做归一化，有一定的疑问，不过对后续影响应该不会特别的大
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        # 接下来是d128和d256 d表示的是卷积核为3，stride为2 IN归一化，Relu激活函数
        # 这里用nn.ModuleList()来写 学习了一下大神的写法
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )

        # 接下来是9个残差网络结构
        # 这个写法好牛逼啊
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # 接下来是两个上采样 u128, u64
        # 复现的是论文里的参数
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
            ]
        )

        # 最后一步
        self.last = nn.Conv2d(num_features,img_channels,kernel_size=7,stride=1,padding=3,padding_mode="reflect")

    def forward(self,x):
        x = self.initial(x)

        # 这是MoudleList前向传播的方式，有点像列表的方式
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


# 接下来对模型做一个测试
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = Generator(img_channels=3).to(DEVICE)
# summary(model, input_size=(3, 256, 256))

