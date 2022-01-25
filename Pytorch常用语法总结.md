## 该内容包含了在代码练习过程中一些常见的Pytorch语法

### 1 torchvision
#### 1.1 torchvision.utils()
    torchvision.utils.make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0)
该函数的作用是将若干幅图像拼接成一幅图像。
其中padding的作用就是子图像与子图像之间的pad有多宽。
nrow表示的是每行和每列放多少张图片，默认为8.
normalize是为了让图片进行归一化，将像素值变为（0，1），默认是不变的。
### 2.