# Cycle GAN 
对论文复现过后的实验结果如下图所示,输入一张现实中的图片，返回一张类似梵高风格的画作：

![real_image](evaluation/real_image.png)![real_image](evaluation/real_image3.png)![real_image](evaluation/real_image7.png)

![van_image](evaluation/y_gen_test.png)![van_image](evaluation/y_gen_test3.png)![van_image](evaluation/y_gen_test6.png)

这个地方要对复现论文的超参数做一个说明，论文中的batch size为1，epoch为200。上述效果只训练了100个epoch，并且剩余100个epoch的学习率要做衰减。