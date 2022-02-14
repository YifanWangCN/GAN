# Image to Image Translation
该方法是对Conditional Gan的变体，即将文本条件变换成图片条件。用一张跟真实原图尺寸大小相同的图片做条件。

数据集链接：https://www.kaggle.com/vikramtiwari/pix2pix-dataset

对论文复现过后的实验结果如下图所示,输入一张卫星俯视图，生成一张对应的google地图：

 ![input_image](evaluation/input_image.png)![Train500_image](evaluation/y_gen_498.png)![target_image](evaluation/target_image.png)

 这个地方要对复现论文的超参数做一个说明，论文中的batch size为1，epoch为200，复现的时候batch size为16，epoch为500 可以看出模型是还需要再训练的。

 # 模型整体模块说明
 整个pix2pix模型一共包含4个主体部分，即生成器，判别器，数据集生成以及模型训练模块。
 在本任务中添加了两个额外的辅助模块，即模型的存储与加载，验证集在的结果以及单独的超参数的设置模块。
 ## 1： 生成器（Generator）
