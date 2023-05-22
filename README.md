# BNDC
这是一个**即插即用**的模块，自适应不同大小的输入
## 用法
在模型的类定义中，按以下方式处理：
* from BNDCModule.py import BNDC
* 在__init__函数中，加入`self.bndc = BNDC(c)`——其中c是此处特征图的通道数
* 在forward函数中，加入`map = self.bndc(map)`
**就大功告成啦**

下面是fine-tune时间，请自由发挥！
