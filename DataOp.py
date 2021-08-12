from PIL import Image
import numpy as np
import cv2
import torch

from Config import CONFIG
import os


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


if __name__ == '__main__':
    config = CONFIG()
    test_image_path = os.path.join(config.BSDS100xN_PATH[2], "img_001_SRF_2_LR.png")
    image = read_image(test_image_path)
    print(type(image), image.shape)

    print(get_all_filenames_in_dir(config.BSDS100xN_PATH[2]))

    image = read_BSDS100(config.BSDS100xN_PATH[2], 21, "LR")
    print(image.shape)
