import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 定义卷积层、批归一化层和激活函数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)  # stride=1 默认值
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 定义快捷连接（shortcut）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 恒等映射

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 模型，增加模型深度
class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层、批归一化和激活函数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个不同层次的 Residual Block，增加每个层的残差块数量
        self.layer1 = self._make_layer(64, num_blocks=3, stride=1)   # 输出尺寸：32x32
        self.layer2 = self._make_layer(128, num_blocks=4, stride=2)  # 输出尺寸：16x16
        self.layer3 = self._make_layer(256, num_blocks=6, stride=2)  # 输出尺寸：8x8
        self.layer4 = self._make_layer(512, num_blocks=3, stride=2)  # 输出尺寸：4x4

        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []

        # 第一个块可能需要下采样
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # 其余的块
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 分类层
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层的权重初始化为 1，偏置为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层的权重初始化
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

