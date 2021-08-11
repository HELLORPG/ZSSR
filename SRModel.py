import torch
import torch.nn as nn
from Config import CONFIG
import DataOp
import numpy as np


class SRModel(nn.Module):

    def __init__(self, config=CONFIG()):
        # 模型初始化
        super(SRModel, self).__init__()

        # 超分辨率比例因子
        self.scale_factor = config.SCALE_FACTOR

        # 上采样层
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        # 卷积层
        self.layer1 = ConvLayer(in_channels=3, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer2 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer3 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer4 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer5 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer6 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer7 = ConvLayer(in_channels=64, out_channels=64, kernel_size=5, padding_mode="zero", stride=1, has_relu=True)
        self.layer8 = ConvLayer(in_channels=64, out_channels=3, kernel_size=5, padding_mode="zero", stride=1, has_relu=False)

    def forward(self, input):
        output = self.upsample(input)

        # 捷径部分
        shortcut = output

        # 残差部分
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        output = self.layer7(output)
        output = self.layer8(output)

        # 合并
        output = output + shortcut

        # 返回
        return output


class ConvLayer(nn.Module):
    """
    一个卷积层，将卷积和 ReLU 合并进行构建，减少最终的代码复用量
    """

    def __init__(self, in_channels: int, out_channels: int, has_relu: bool, kernel_size=5, stride=1, padding=2, padding_mode="reflection"):
        super(ConvLayer, self).__init__()

        if padding_mode == "zero":
            self.pad = nn.ZeroPad2d(padding)
        elif padding_mode == "reflection":
            self.pad = nn.ReflectionPad2d(padding)
        elif padding_mode == "replication":
            self.pad = nn.ReplicationPad2d(padding)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)

        # 没有加入BN层

        self.relu = nn.ReLU(inplace=True)

        self.has_relu = has_relu

    def forward(self, input):
        output = self.pad(input)
        output = self.conv(output)

        if self.has_relu is True:
            output = self.relu(output)

        return output


if __name__ == '__main__':
    config = CONFIG()
    image = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], 1, "LR")
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # print(image.shape)
    # # print(image)
    #
    # sr = SRModel(config=config)
    # image = np.transpose(image, (0, 3, 1, 2))
    # image = torch.from_numpy(image)
    # image = image.float()
    # print(image.shape)
    # image = torch.tensor(image)

    sr = SRModel(config=config)
    image = DataOp.ndarray2torch(image)

    print(image.shape)
    sr_output = sr(image)
    print(sr_output.shape)

