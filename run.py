import DataOp

from System import pytorch_version
import platform
import os
from Config import CONFIG
from SRModel import SRModel


def print_system():
    """
    输出当前的运行环境检测
    :return:
    """
    print(">>>>  Pytorch Version is %s" % pytorch_version())
    print(">>>>  System is %s" % platform.system().lower())


def run_bsds100_a_image(model: SRModel, index: int, config=CONFIG()):
    lr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "LR")
    # print(lr_im.shape)
    hr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "HR")

    model.train_net(lr_im, hr_im)


if __name__ == '__main__':
    print(">>>>  ZSSR run.py begin")
    config = CONFIG()   # 获得基准的config配置

    sr_model = SRModel(config)

    run_bsds100_a_image(sr_model, 2, config=config)


