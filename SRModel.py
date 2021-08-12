import torch
import torch.nn as nn
from Config import CONFIG
import DataOp
import numpy as np
import cv2
from PIL import Image


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

        # Loss函数
        self.loss_func = nn.L1Loss()

        # 定义网络的优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)

        # Something Else
        self.init_epoch = config.INIT_EPOCH
        self.current_lr = config.LEARN_RATE
        self.min_lr = config.MIN_LR
        self.print_train_epoch = config.PRINT_TRAIN_EPOCH

        self.device =  config.DEVICE

        # 数据的设置
        self.input_size = config.NET_IMAGE_SIZE

        self.to(self.device)


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

    def _train(self, lr_im, hr_im, epoch):
        self.optim.zero_grad()

        sr_output = self.forward(lr_im)
        loss = self.loss_func(sr_output, hr_im)
        loss.backward()

        self.optim.step()

        if self.print_train_epoch:
            print(">>>> Train Epoch %d, Loss=%f, Lr=%f" % (epoch, loss, self.current_lr))

        return

    def train_net(self, lr_im, hr_im):
        # ZSSR网络使用的训练并不是固定Epoch的，而是以训练过程中的lr动态调整作为判定依据的
        # 如果在lr的动态递减过程中，lr低于某一个值，则停止学习
        epoch = self.init_epoch
        while True:
            self.current_lr = self.optim.state_dict()['param_groups'][0]['lr']   # 获取当前的学习率
            if self.current_lr < self.min_lr:
                break

            # 进行正常的训练过程
            epoch += 1

            # 获得训练需要使用的数据
            lr_data, hr_data = DataOp.get_train_images(lr_im, self.input_size, self.scale_factor)

            self._train(lr_data.to(self.device), hr_data.to(self.device), epoch)




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
    lr_image = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], 21, "LR")
    hr_image = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], 21, "HR")

    # lr_image = lr_image.reshape((1, lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]))
    # hr_image = hr_image.reshape((1, hr_image.shape[0], hr_image.shape[1], hr_image.shape[2]))

    model = SRModel(config)

    # lr_images, hr_images = DataOp.get_train_images(lr_image, config.NET_IMAGE_SIZE, config.SCALE_FACTOR)
    # print(lr_images.shape)

    model.train_net(lr_image, hr_image)

