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


def run_bsds100_a_image(index: int, config=CONFIG()):
    model = SRModel(config)

    lr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "LR")
    # print(lr_im.shape)
    hr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "HR")

    model.train_net(lr_im, hr_im)


if __name__ == '__main__':
    print(">>>>  ZSSR run.py begin")
    print_system()
    config = CONFIG()   # 获得基准的config配置

    if config.system == "linux":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID

    for i in range(0, 3):
        run_bsds100_a_image(2, config=config)
    # index=2, 30.166222309772564 0.9432615378365822


