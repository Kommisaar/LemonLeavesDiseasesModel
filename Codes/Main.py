from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_M_Weights

from LemonLeavesDiseasesModel import LemonLeavesDiseasesModel
from LemonLeavesDiseasesDataset import LemonLeavesDiseasesDataset, lemon_leaves_load_data
from LemonLeavesDiseasesTrain import lemon_leaves_train_model


def get_lemon_transformer():
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
    return transform


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化预训练模型
    model = LemonLeavesDiseasesModel(weights=EfficientNet_V2_M_Weights.DEFAULT).to(device)

    # 加载数据
    root_dir = Path('../Data/Original Dataset')
    data, label = lemon_leaves_load_data(root_dir)

    # 数据集划分
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(data, label, test_size=0.2,
                                                                                random_state=42, stratify=label)
    train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels,
                                                                      test_size=0.25, random_state=42, stratify=label)
    # 定义训练轮次、批次大小、学习率、惩罚率
    num_epochs = 75
    learning_rate = 0.001
    train_batch_size = 128
    val_batch_size = 512

    lemon_leaves_transformer = get_lemon_transformer()

    # 数据加载器
    train_loader = DataLoader(
        LemonLeavesDiseasesDataset(train_data, train_labels, transform=lemon_leaves_transformer),
        batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(
        LemonLeavesDiseasesDataset(val_data, val_labels),
        batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(
        LemonLeavesDiseasesDataset(test_data, test_labels),
        batch_size=val_batch_size, shuffle=False)

    # 获取类型名称
    class_types = [str(class_name.stem) for class_name in root_dir.iterdir()]

    continue_last_model = True
    if continue_last_model:
        model.load_state_dict(torch.load("../ModelWeights/LastModel-LemonLeaves.pth"))

    # 定义损失函数
    criterion = CrossEntropyLoss()

    # 定义优化器
    # 定义只需要训练的分类层参数
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(params, lr=learning_rate)

    # 训练模型
    model = lemon_leaves_train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, class_types,
                                     device)

    # 保存模型
    torch.save(model.state_dict(), "../ModelWeights/LastModel-LemonLeaves.pth")
