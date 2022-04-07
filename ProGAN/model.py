import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2   # 它用于获取一个数字的以 2 为底的对数，它接受一个数字并返回给定数字的以 2 为底的对数

# 需要申明的是在该论文中，作者提到了几种trick
# EQUALIZED LEARNING RATE(个人感觉是对参数的动态均衡）
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):  #该默认值保证特征图的尺寸并没有减少
        super().__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)

        # 该方法类似何凯明初始化的方法，in_channels其实代表的就是上一层的节点数
        self.scale = (gain / (in_channels * kernel_size * kernel_size)) ** 0.5

        # 需要注意的是，我们不希望 conv里的bias被scale，我们只scale 卷积核权重参数
        # 先将参数赋值给对象 self.bias 相当于copy, 然后再将conv.bias变成None（removing the bias)
        self.bias = self.conv.bias
        self.conv.bias = None

        # 这个时候对权重先做一个论文中平凡的初始化，即正太分布初始化, 偏差bias设为0即可
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):

        # 实际上是对权重进行scale 但是也可以通过参数的方式传给权重
        return self.conv(x * self.scale) + self.bias.view(1,self.bias.shape[0],1,1)


# PIXELWISE FEATURE VECTOR NORMALIZATION
# 这是论文中提到的第二个小trick
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

        # 论文中的设定
        self.epsilon = 1e-8

    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True) + self.epsilon)



# 接下来就是构建判别器或者生成器里的每一个卷积Block了
class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels,out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self,x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

# 论文中生成器中的特征图节点数的取值
# 该生成器其实有9个block
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

# 下面就是生成器模型的构成了
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()

        # 这是网络的第一层，该层有一个反卷积，将1*1的维度 变成4*4的维度
        self.initial = nn.Sequential(
            PixelNorm(), # 一个小trick
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1,padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        # 用于后续的fade_in层
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1,stride=1,padding=0)

        # 下面这个写法非常之精妙
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        for i in range(len(factors)-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c,img_channels,kernel_size=1, stride=1, padding=0))

    # 该过程用于平滑过度 分辨率加倍的情况。 有点类似于残差网络的结构
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)


    # steps 这个参数表示维度的翻倍数，比如step=0，4*4,step=1, 8*8
    def forward(self, x, alpha, steps):
        out = self.initial(x) # 4*4

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest") # 上采样
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    # 这个地方的 in_channels 依然是 512
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()

        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers =  nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # i 的取值是从 8 开始， 8,7,6 >> 1
        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])

            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c,use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_c,kernel_size=1,stride=1,padding=0))


        self.initial_rgb = WSConv2d(img_channels,in_channels, kernel_size=1,stride=1,padding=0)
        self.rgb_layers.append(self.initial_rgb)

        # down sampling using avg pool
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4,stride=1,padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled


    def minibatch_std(self,x):
        batch_statistics = torch.std(x,dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        # 这也是为什么最后是从512个特征图变成了513个特征图
        return torch.cat([x, batch_statistics], dim=1)

    # 这里输入的x一定是一张3个通道的图片
    def forward(self,x,alpha,steps): # steps 0 4*4 1 8*8 steps=8 代表的是1024*1024
        cur_step = len(self.prog_blocks) - steps

        # from RGB
        out = self.leaky(self.rgb_layers[cur_step](x))


        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0],-1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

# test
if __name__ == "__main__":
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(Z_DIM, IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")




