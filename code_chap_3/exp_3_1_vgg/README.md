## 实验目的

  掌握卷积神经网络的设计原理，掌握卷积神经网络的使用方法，能够使用 Python 语言实现VGG19网络模型对给定的输入图像进行分类。

  具体包括：

```
a) 加深对深度卷积神经网络中卷积层、最大池化层等基本单元的理解。
b) 利用Python语言实现VGG19的前向传播计算，加深对VGG19网络结构的理解，为后续风格迁移中使用 VGG19 网络计算风格损失奠定基础。
c) 在第2.1节实验的基础上将三层神经网络扩展为 VGG19 网络，加深对神经网络工程 实现中基本模块演变的理解，
   为后续建立更复杂的综合实验(如风格迁移)奠定基础。
```

  实验进程：10%。

  实验工作量：约 300 行代码，约需 3 个小时。



## 实验环境

  硬件环境：CPU。

  软件环境：Python 编译环境及相关的扩展库，包括 Python 2.7.9，Pillow 3.4.2，SciPy 0.18.1，NumPy 1.11.2(本实验不需使用 TensorFlow 等深度学习框架)。

  数据集　：官方训练 VGG19 使用的数据集为 ImageNet。

​         该数据集包括约 128 万训练 图像和 5 万张验证图像，共有 1000 个不同的类别。本实验使用了官方训练好的模型参数， 

​         并不需要直接使用 ImageNet 数据集进行 VGG19 模型的训练。

  

实验内容和步骤

  详情请查看实验指导书 