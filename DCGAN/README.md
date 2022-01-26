# 反卷积（转置卷积）

 该概念是本任务的最终要的概念，通过反卷积的方式生成图像。

要明确反卷积的步骤
1：对输入的feature map进行padding操作，得到的是新的feature map。
2： 随机初始化卷积核的参数
3： 用修改后的参数对新的feature map进行卷积操作

### 1. 首先是 S=1 的情况下的反卷积

举DCGAN生成器中的第一步，由[100,1,1]维度的噪声，生成[1024,4,4]维度的特征图。我们需要关注的是如何将一个 1*1 维度的特征图，通过反卷积生成为一个 4*4 维度的特征图

    有几个参数是需要明确的：
    kernel size = 4
    stride = 1
    padding = 0

通过上述步骤，我们复现一下在这种情况下是如何转变的
1：对输入的 1*1 特征图进行padding 因为padding在这里为0，所以该 1*1 特征图的维度保持不变
2: 对用于反卷积的Kernel进行初始化
3: 对新的图像进行，kernel size = 4， stride = 1，new_padding = K - padding - 1 = 3 的卷积。
    在该过程里，通过维度计算公式，我们可以得到
    {[1 + （2 * 3） - 4]/1} + 1 = 4。