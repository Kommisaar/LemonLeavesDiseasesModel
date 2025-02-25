import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_v2_m


class LemonLeavesDiseasesModel(nn.Module):
    def __init__(self, num_classes: int, weights=None):
        super().__init__()
        # 加载预训练权重
        self.model = efficientnet_v2_m(weights=weights)

        # 冻结特征提取层
        for param in self.model.parameters():
            param.requires_grad = False

        # 替换分类器
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(1280, num_classes)
        )

        # 初始化分类器权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for module in self.model.classifier:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入的图像，大小480x480。
        :return: 预测的标签
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            output = self.model(x)
            pred = torch.argmax(output, dim=1)
        return pred

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        :param x: 输入的图像，大小480x480。
        :return: 预测的概率
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            output = self.model(x)
            pred = torch.softmax(output, dim=1)
        return pred


