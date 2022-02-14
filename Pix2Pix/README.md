# Image to Image Translation
该方法是对Conditional Gan的变体，即将文本条件变换成图片条件。用一张跟真实原图尺寸大小相同的图片做条件。

数据集链接：https://www.kaggle.com/vikramtiwari/pix2pix-dataset

对论文复现过后的实验结果如下图所示,输入一张卫星俯视图，生成一张对应的google地图：
 ![input_image](evaluation/input_image.png)![Train500_image](evaluation/y_gen_498.png)![target_image](evaluation/target_image.png)