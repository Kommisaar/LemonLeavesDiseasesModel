# 项目介绍

## 项目名称

基于`EfficientNetV2.M`卷积神经网络实现柠檬叶疾病（`Lemon Leaf Disease`）区分

## 项目描述

本项目旨在利用深度学习中的卷积神经网络（CNN）技术，构建一个高效、精准的柠檬叶片疾病自动识别系统。针对农业生产中柠檬叶片病害（如疮痂病、黄斑病、炭疽病等）人工诊断效率低、误判率高等痛点，通过计算机视觉技术实现病害的快速分类。

# 数据集

## 数据集介绍

柠檬叶病数据集（`LLDD`）是一个高质量的图像数据集，旨在用于训练和评估柠檬叶病分类的机器学习模型。该数据集包含健康和患病柠檬叶的图像，使其适用于植物疾病检测、图像分类以及农业中的深度学习应用。

## 数据集来源

数据集来源于:https://www.kaggle.com/datasets/mahmoudshaheen1134/lemon-leaf-disease-dataset-lldd/data

## 数据增强

使用数据增强来扩展数据集，提升模型的泛化能力。

```python
import torchvision.transforms.v2 as v2

transform = v2.Compose([
    # 调整大小
    v2.Resize(size=(480, 480)),
    # 几何变换
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-30, 30)),
    v2.CenterCrop(size=(480, 480)),
    # 颜色变换
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    v2.RandomGrayscale(p=0.2),
    # 转换为张量
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    # 归一化
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 数据分割

将数据集分割为**训练集**、**验证集**以及**测试集**，按照`6:2:2`的比例划分。

```python
from sklearn.model_selection import train_test_split

# 数据集划分
train_val_data, test_data, train_val_labels, test_labels = train_test_split(data, label, test_size=0.2,
                                                                            random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels,
                                                                  test_size=0.25, random_state=42)
```

