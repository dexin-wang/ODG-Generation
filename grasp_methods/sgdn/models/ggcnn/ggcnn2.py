import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNN2(nn.Module):
    def __init__(self, angle_cls, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            # filter_sizes = [16,  # First set of convs
            #                 16,  # Second set of convs
            #                 32,  # Dilated convs
            #                 16]  # Transpose Convs
            
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            32]  # Tran

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # ============= 到这里，和sgdn_cover的网络结构一样 =============

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.point_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.angle_output = nn.Conv2d(filter_sizes[3], angle_cls, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], angle_cls, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, x):
        x = self.features(x)

        point_output = self.point_output(x)
        angle_output = self.angle_output(x)
        width_output = self.width_output(x)

        return point_output, angle_output, width_output


class GGCNN3(nn.Module):
    """
    相比于ggcnn2
    (1) 增加 BN
    (2) 增加 dropout
    (3) 预测head由一个卷积增加为两个卷积
    """
    def __init__(self, angle_cls, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16, 32, 64, 64, 32] 

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.BatchNorm2d(filter_sizes[3]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.BatchNorm2d(filter_sizes[3]),
            nn.ReLU(inplace=True),
        )

        # 方案1
        # self.point_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        # self.angle_output = nn.Conv2d(filter_sizes[3], angle_cls, kernel_size=1)
        # self.width_output = nn.Conv2d(filter_sizes[3], angle_cls, kernel_size=1)

        # 方案2
        self.point_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter_sizes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(filter_sizes[4], 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(16, 1, kernel_size=1, stride=1)
        )

        self.angle_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter_sizes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(filter_sizes[4], 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(32, angle_cls, kernel_size=1, stride=1)
        )

        self.width_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter_sizes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(filter_sizes[4], 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(32, angle_cls, kernel_size=1, stride=1)
        )


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        


    def forward(self, x):
        x = self.features(x)

        point_output = self.point_output(x)
        angle_output = self.angle_output(x)
        width_output = self.width_output(x)

        return point_output, angle_output, width_output