from PIL import Image
import numpy as np
import cv2
import torch

from Config import CONFIG
import os
import random
from torchvision.transforms import functional as F


def read_image(filepath: str) -> np.ndarray:
    """
    :param filepath: 图像的路径
    :return: 图像的 ndarray 形式，(h, w, c)
    """
    image = cv2.imread(filename=filepath, flags=None)

    return image


def get_all_filenames_in_dir(dirpath: str) -> list:
    """
    获取某一个文件夹下的所有文件名
    :param dirpath:
    :return:
    """
    filenames = [filename for filename in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, filename)) is False]

    return filenames


def read_BSDS100(dataset_path: str, index: int, type: str):
    """
    读取一条 BSDS100 中的图像
    :param dataset_path: 数据集的存放路径
    :param index: 图像的编号
    :param type: 例如 LR 或者 HR 等字符串格式的标识符，用来读取该编号图像的不同版本
    :return:
    """
    filename = str(index)

    if index < 10:
        filename = "00" + filename
    elif index < 100:
        filename = "0" + filename
    else:
        filename = filename

    filename = "img_" + filename + "_SRF_" + dataset_path[-1] + "_" + type + ".png"

    return read_image(os.path.join(dataset_path, filename))


def ndarray2tensor(data):
    """
    将opencv读取到的格式为(n,h,w,c)的数组转换为(n,c,h,w)
    :param data: opencv读到的ndarray数据
    :return: 适用输入torch网络的数据
    """

    data = np.transpose(data, (0, 3, 1, 2))
    # data = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))

    data = torch.from_numpy(data).float()

    return data


def show_ndarray_image(im):
    """
    展示一张以(h,w,c)存储的ndarray图像
    :param image: ndarray which shape is (h,w,c)
    :return:
    """
    im = Image.fromarray(im)
    im.show()
    return


def get_hr_father(im: np.ndarray, min_size: int, scale_factor) -> np.ndarray:
    """
    对输入的图像进行下采样，获得论文中的HR-Father
    :param im: 输入的图像(h,w,c)
    :param min_size: 输出图像的最小尺寸
    :return:
    """
    # print(im.shape)

    # 获得一个降采样的因子
    # TODO 这里应该会有更好的采样方式，原论文中强调，这里原文中强调，约接近原图大小的HR-Father将更有概率被选中，也即是一个不均匀的采样，而我这里的采样是均匀的，其实是有一定的偏差的。
    downsample_factor = random.uniform(0.8, 1.0)

    h, w = im.shape[0], im.shape[1]
    h1, w1 = round(h * downsample_factor), round(w * downsample_factor)

    if h1 < min_size * scale_factor or w1 < min_size * scale_factor:
        # 进行修正，这样的系数是不行的
        if h < w:
            # 此时的修正以h为主
            downsample_factor = min_size * scale_factor / h
        else:
            downsample_factor = min_size * scale_factor / w

    h1, w1 = round(h * downsample_factor), round(w * downsample_factor)
    # print(h1, w1)

    hr_father = cv2.resize(im, dsize=(w1, h1), interpolation=cv2.INTER_CUBIC)   # 这个函数非常奇怪，反而又是以(w,h)作为尺寸的格式了

    return hr_father


def get_train_images(im: np.ndarray, size: int, scale_factor):
    """
    通过一张输入图像，获得数据增强之后的网络训练输入图像群（8张）
    :param im: 输入的一张图像，应该是一张
    :param size: 最终输入网络的图像尺寸
    :return:
    """
    hr_fathers = []

    hr_father = get_hr_father(im, size, scale_factor)

    # HR-Father拷贝成为八份
    for i in range(0, 8):
        hr_fathers.append(hr_father.copy())

    # 进行旋转翻转等变换
    for i in range(0, 8):
        hr_fathers[i] = np.rot90(hr_fathers[i], int(i/2))
        if i % 2 == 1:
            hr_fathers[i] = np.flipud(hr_fathers[i])

    # 对HR-Fathers进行裁剪
    for i in range(0, 8):
        hr = hr_fathers[i]
        h, w = hr.shape[0], hr.shape[1]
        crop_size = size * scale_factor
        h1, w1 = h - crop_size, w - crop_size
        crop_h = random.randint(0, h1)
        crop_w = random.randint(0, w1)

        hr_fathers[i] = hr[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, ...]

    # for hr in hr_fathers:
    #     show_ndarray_image(hr)
    # show_ndarray_image(hr_fathers[1])

    lr_sons = []

    for hr_father in hr_fathers:
        lr_son = cv2.resize(hr_father, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        lr_sons.append(lr_son)

    # show_ndarray_image(lr_sons[0])
    # print(F.to_tensor(hr_fathers[0]).shape)

    # for i in range(0, 8):
    #     hr_fathers[i] = hr_fathers[i].reshape((1, hr_fathers[i].shape[0], hr_fathers[i].shape[1], hr_fathers[i].shape[2]))
    #     lr_sons[i] = lr_sons[i].reshape((1, lr_sons[i].shape[0], lr_sons[i].shape[1], lr_sons[i].shape[2]))

    # x = np.ones((2, 2, 3), dtype=np.uint8)
    # print(type(x))
    # x[1,1,2] = 10
    # x = F.to_tensor(x)
    # print(x)

    lr_sons, hr_fathers = totensor_lr_hr(lr_sons, hr_fathers)
    # print(lr_sons[0].shape)

    lr_data = torch.stack((
        lr_sons[0],
        lr_sons[1],
        lr_sons[2],
        lr_sons[3],
        lr_sons[4],
        lr_sons[5],
        lr_sons[6],
        lr_sons[7]
    ), dim=0)

    hr_data = torch.stack((
        hr_fathers[0],
        hr_fathers[1],
        hr_fathers[2],
        hr_fathers[3],
        hr_fathers[4],
        hr_fathers[5],
        hr_fathers[6],
        hr_fathers[7]
    ), dim=0)

    lr_data = (lr_data - 0.5) / 0.5
    hr_data = (hr_data - 0.5) / 0.5

    # print(lr_data)

    # return ndarray2tensor(lr_data), ndarray2tensor(hr_data)
    return lr_data, hr_data

def totensor_lr_hr(lr: list, hr: list):
    for i in range(0, len(lr)):
        lr[i] = F.to_tensor(lr[i].astype(np.uint8))
        hr[i] = F.to_tensor(hr[i].astype(np.uint8))

    return lr, hr


if __name__ == '__main__':
    config = CONFIG()
    test_image_path = os.path.join(config.BSDS100xN_PATH[2], "img_001_SRF_2_LR.png")
    image = read_image(test_image_path)
    print(type(image), image.shape)

    # print(get_all_filenames_in_dir(config.BSDS100xN_PATH[2]))

    image = read_BSDS100(config.BSDS100xN_PATH[2], 21, "LR")

    lr, hr = get_train_images(image, 64, 2)
    print(lr.shape)

    # print(image.shape)
