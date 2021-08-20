import os
import math

import skimage.transform
import torch
import torch.nn as nn
from Config import CONFIG
import DataOp
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



class SRModel(nn.Module):

    def __init__(self, config=CONFIG()):
        # 模型初始化
        super(SRModel, self).__init__()

        # 超分辨率比例因子
        self.scale_factor = config.SCALE_FACTOR

        # 上采样层
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        # 卷积层
        self.layer1 = ConvLayer(in_channels=3, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer2 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer3 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer4 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer5 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer6 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer7 = ConvLayer(in_channels=64, out_channels=64, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=True)
        self.layer8 = ConvLayer(in_channels=64, out_channels=3, kernel_size=config.KERNEL_SIZE, padding_mode=config.PADDING_MODE, stride=1, padding=config.PADDING, has_relu=False)

        # Loss函数
        self.loss_func = nn.L1Loss()
        # self.loss_func = nn.MSELoss()
        # 定义网络的优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)

        # Something Else
        self.init_epoch = config.INIT_EPOCH
        self.max_epoch = config.MAX_EPOCH
        self.current_lr = config.LEARN_RATE
        self.min_lr = config.MIN_LR
        self.print_train_epoch = config.PRINT_TRAIN_EPOCH
        self.print_eval_epoch = config.PRINT_EVAL_EPOCH
        self.loss_neighbor = []
        self.loss_neighbor_len = config.LOSS_NEIGHBOR_LEN
        self.lr_drop_when = config.LR_DROP_WHEN
        self.lr_drop_rate = config.LR_DROP_RATE
        self.has_normalize = config.HAS_NORMALIZE

        # 优化器更新逻辑
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 1, gamma=1 / self.lr_drop_rate, last_epoch=-1)

        self.device =  config.DEVICE

        # 数据的设置
        self.input_size = config.NET_IMAGE_SIZE
        self.mean = None
        self.std = None

        # 调试输出
        self.index = config.INDEX
        timestamp = time.strftime("%Y_%m_%d+%H_%M_%S", time.localtime())
        if self.index is None:
            tb_path = os.path.join(os.path.join(config.PROJECT_PATH, config.TB_LOG_DIR), timestamp)
        else:
            tb_path = os.path.join(os.path.join(config.PROJECT_PATH, config.TB_LOG_DIR), "No." + str(self.index) + "_" + timestamp)
        # os.mkdir(tb_path)
        self.tb_writer = SummaryWriter(tb_path)
        self.eval_every_epoch = config.EVAL_EVERY_EPOCH

        # 结果增强
        self.has_back_projection = config.BACK_PROJECTION
        self.back_projection_times = config.BACK_PROJECTION_TIMES
        self.combine_method = config.COMBINE_METHOD

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
        self.train()    # 进入训练模式

        self.optim.zero_grad()

        sr_output = self.forward(lr_im)
        loss = self.loss_func(sr_output, hr_im)
        loss.backward()

        self.optim.step()

        if self.print_train_epoch:
            print(">>>> Train Epoch %d, Loss=%f, Lr=%f" % (epoch, loss, self.current_lr))

        self.tb_writer.add_scalar("Train Loss", loss, global_step=epoch)
        self.tb_writer.add_scalar("Learn Rate", self.current_lr, epoch)

        return loss

    def train_net(self, lr_im, hr_im):
        """
        训练整个网络
        :param lr_im: 数据集中的LR Image
        :param hr_im: 数据集中的HR Image，这部分不用于训练，只用于评估网络
        :return:
        """
        # ZSSR网络使用的训练并不是固定Epoch的，而是以训练过程中的lr动态调整作为判定依据的
        # 如果在lr的动态递减过程中，lr低于某一个值，则停止学习
        epoch = self.init_epoch
        self.mean, self.std = DataOp.get_mean_std(lr_im)
        while True:
            self.current_lr = self.optim.state_dict()['param_groups'][0]['lr']   # 获取当前的学习率
            if self.current_lr < self.min_lr or epoch >= self.max_epoch:
                # TODO 这里可以加入一个处理模型结束的函数
                # return self.evaluate_net(lr_im=lr_im, hr_im=hr_im, epoch=epoch+1)
                return self.final_evaluate(lr_im, hr_im, epoch)
                # break

            # 进行正常的训练过程
            epoch += 1

            # 获得训练需要使用的数据
            lr_data, hr_data = DataOp.get_train_images(lr_im, self.input_size, self.scale_factor, has_normalize=self.has_normalize)

            loss = self._train(lr_data.to(self.device), hr_data.to(self.device), epoch)
            if self.need_drop_lr(loss):
                self.scheduler.step()
            if self.eval_every_epoch:
                self.evaluate_net(lr_im, hr_im, epoch=epoch)
        return

    def _test(self, lr_im):
        self.eval()
        with torch.no_grad():
            sr_im = self.forward(lr_im)
        return sr_im

    def evaluate_net(self, lr_im, hr_im, epoch=0) -> [float, float]:
        """
        训练过程中的评估函数，并没有对八个结果中的像素取中位数，相当于没有进行这一步增强。
        :param lr_im: 数据集中的LR图像
        :param hr_im: 数据集中的HR图像
        :param epoch: 当前的轮数
        :return:
        """
        lr_copy = lr_im
        hr_copy = hr_im
        lr_im = F.to_tensor(lr_im)
        hr_im = F.to_tensor(hr_im)
        lr_im = lr_im.reshape((1, lr_im.shape[0], lr_im.shape[1], lr_im.shape[2]))
        hr_im = hr_im.reshape((1, hr_im.shape[0], hr_im.shape[1], hr_im.shape[2]))

        if self.has_normalize:
            # lr_im = (lr_im - 0.5) / 0.5
            lr_im = F.normalize(lr_im, self.mean, self.std)

        sr_im = self._test(lr_im.to(self.device))

        sr_im = sr_im.cpu().numpy().reshape((sr_im.shape[1], sr_im.shape[2], sr_im.shape[3]))

        if self.has_normalize:
            # sr_im = sr_im * 0.5 + 0.5
            sr_im = DataOp.de_normalize(sr_im, self.mean, self.std)

        sr_im = np.transpose(sr_im, (1, 2, 0))

        hr_im = hr_im.cpu().numpy().reshape((hr_im.shape[1], hr_im.shape[2], hr_im.shape[3]))
        hr_im = np.transpose(hr_im, (1, 2, 0))

        sr_im = cv2.cvtColor(sr_im, cv2.COLOR_BGR2RGB)
        hr_im = cv2.cvtColor(hr_im, cv2.COLOR_BGR2RGB)

        # sr_im *= 255
        # sr_im = np.clip(sr_im, 0, 255)

        psnr = peak_signal_noise_ratio(hr_im, sr_im)
        ssim = structural_similarity(hr_im, sr_im, multichannel=True)
        # print(psnr, ssim)
        if self.print_eval_epoch:
            print(">>>>>>  Test Epoch %d: PSNR=%f, SSIM=%f" % (epoch, psnr, ssim))
        else:
            if epoch % 100 == 0:
                print(">>>>>>  Test Epoch %d: PSNR=%f, SSIM=%f" % (epoch, psnr, ssim))
        self.tb_writer.add_scalar("PSNR", psnr, global_step=epoch)
        self.tb_writer.add_scalar("SSIM", ssim, global_step=epoch)

        # if epoch % 100 == 0:
        #     print(self.final_evaluate(lr_copy, hr_copy, epoch))

        return psnr, ssim


    def final_evaluate(self, lr_im: np.ndarray, hr_im: np.ndarray, epoch: int, save_img=False):
        """
        进行一个综合的评估，这次评估包括了对结果数据的增强等等。
        :param lr_im:
        :param hr_im:
        :param epoch:
        :param save_img: 是否保存最终SR之后的图像
        :return:
        """
        lr_images = []  # 存储经过数据增强之后的八张图像

        # 进行数据增强，通过旋转和镜像得到8张图像
        for i in range(0, 8):
            lr_images.append(np.rot90(lr_im, int(i / 2)))
            if i % 2 == 1:  # 此时进行镜像
                lr_images[i] = np.flipud(lr_images[i])

        # 转换成为Tensor
        for i in range(0, 8):
            lr_images[i] = F.to_tensor(lr_images[i].copy())

        # 是否进行标准化
        if self.has_normalize:
            for i in range(0, 8):
                lr_images[i] = F.normalize(lr_images[i], self.mean, self.std)

        # 循环训练得到所有的SR输出结果
        sr_images = []
        for i in range(0, 8):
            sr_images.append(self._test(lr_images[i].reshape((1, lr_images[i].shape[0], lr_images[i].shape[1], lr_images[i].shape[2])).to(self.device)))
            # 输出的sr_images是(1,c,h,w)格式
            sr_images[i] = sr_images[i].reshape((sr_images[i].shape[1], sr_images[i].shape[2], sr_images[i].shape[3]))

        # 将Tensor的输出转换为Numpy
        for i in range(0, 8):
            sr_images[i] = sr_images[i].cpu().numpy()

        # 进行逆标准化
        if self.has_normalize:
            for i in range(0, 8):
                sr_images[i] = DataOp.de_normalize(sr_images[i], mean=self.mean, std=self.std)

        # 更改成为(h,w,c)
        for i in range(0, 8):
            sr_images[i] = np.transpose(sr_images[i], (1, 2, 0))

        # 对HR图像进行相似的处理
        hr_im = F.to_tensor(hr_im)  # 归一化
        hr_im = hr_im.cpu().numpy()     # 转换成为Numpy
        hr_im = np.transpose(hr_im, (1, 2, 0)) # 调整为(h,w,c)

        # 对颜色系统进行调整
        for i in range(0, 8):
            sr_images[i] = cv2.cvtColor(sr_images[i], cv2.COLOR_BGR2RGB)
        hr_im = cv2.cvtColor(hr_im, cv2.COLOR_BGR2RGB)

        # 对之前旋转和镜像进行复原
        for i in range(0, 8):
            if i % 2 == 1:
                sr_images[i] = np.flipud(sr_images[i])
            sr_images[i] = np.rot90(sr_images[i], -int(i/2))

        # 对八种方向和镜像得到的SR图像进行合并处理
        sr_im = np.zeros_like(sr_images[0])
        for h in range(0, sr_images[0].shape[0]):
            for w in range(0, sr_images[0].shape[1]):
                for c in range(0, sr_images[0].shape[2]):
                    x = []
                    for i in range(0, 8):
                        x.append(sr_images[i][h,w,c])
                    if self.combine_method == "median":
                        x_new = np.median(x)
                    else:
                        x_new = np.mean(x)
                    sr_im[h,w,c] = x_new

        # 对SR进行迭代修正
        if self.has_back_projection:
            lr_im = F.to_tensor(lr_im.copy())
            lr_im = lr_im.cpu().numpy()
            lr_im = np.transpose(lr_im, (1, 2, 0))
            lr_im = cv2.cvtColor(lr_im, cv2.COLOR_BGR2RGB)
            sr_im = self.back_projection(lr_im, sr_im, self.scale_factor, self.back_projection_times)

        # 对SR进行(0,1)修正
        sr_im = np.clip(sr_im, 0, 1)

        # 评估SR和HR的差距
        psnr = peak_signal_noise_ratio(hr_im, sr_im)
        ssim = structural_similarity(hr_im, sr_im, multichannel=True)

        # save_img = True
        if save_img:
            # 如果需要保存图像
            sr_im = cv2.cvtColor(sr_im, cv2.COLOR_RGB2BGR)
            # sr_im = int(sr_im * 256)
            sr_im = np.multiply(sr_im, 256)
            sr_im = sr_im.astype(int)
            cv2.imwrite("output/test.png", sr_im)


        return psnr, ssim

    def back_projection(self, lr_im: np.ndarray, sr_im: np.ndarray, scale_factor, times: int):
        """
        数据增强
        :param im: 需要增强的图像
        :param scale_factor: 缩放的倍数
        :param times: 进行增强的次数
        :return:
        """
        # h, w = im.shape[0], im.shape[1]
        # h1, w1 = round(h/scale_factor), round(w/scale_factor)
        for i in range(0, times):
            sr_im += cv2.resize(lr_im -
                                cv2.resize(sr_im, (lr_im.shape[1], lr_im.shape[0])), (sr_im.shape[1], sr_im.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return sr_im

    def need_drop_lr(self, current_loss) -> bool:
        if len(self.loss_neighbor) < self.loss_neighbor_len:
            self.loss_neighbor.append(current_loss.to("cpu").item())
            return False
        else:
            del self.loss_neighbor[0]
            self.loss_neighbor.append(current_loss.to("cpu").item())

            x = list(range(0, self.loss_neighbor_len))
            # for i in range(0, len(x)):
            #     x[i] *= self.current_lr
            # print(x)
            # print(x, self.loss_neighbor)

            # 这是一种自己的算法
            # a = np.polyfit(x, self.loss_neighbor, 1)
            # k = abs(np.poly1d(a)[1])
            # std = np.std(self.loss_neighbor, ddof=1)

            # 替换成为原论文中采用的算法
            [k, _], [[var, _], _] = np.polyfit(x, self.loss_neighbor, 1, cov=True)

            std = math.sqrt(var)
            k = abs(k)

            # print(std, np.std(self.loss_neighbor))
            # print(k)

            if std > k * self.lr_drop_when:
                self.loss_neighbor.clear()
                return True
            else:
                return False




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
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.has_relu = has_relu

    def forward(self, input):
        output = self.pad(input)
        output = self.conv(output)

        # output = self.bn(output)

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

    model.train_net(lr_image, hr_image)

