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
    # config.INDEX = index    # 确定当前的index
    model = SRModel(config)

    lr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "LR")
    # print(lr_im.shape)
    hr_im = DataOp.read_BSDS100(config.BSDS100xN_PATH[2], index, "HR")

    return model.train_net(lr_im, hr_im)

def run_bsds100_images():
    psnr_sum, ssim_sum = 0.0, 0.0
    for i in range(1, 101):
        config = CONFIG()
        config.INDEX = i
        psnr, ssim = run_bsds100_a_image(i, config)
        print("No. %d: PSNR=%f, SSIM=%f" % (i, psnr, ssim))
        psnr_sum += psnr
        ssim_sum += ssim

    print("Total Images: PSNR=%f, SSIM=%f" % (psnr_sum/100, ssim_sum/100))

    return psnr_sum/100, ssim_sum/100


if __name__ == '__main__':
    print(">>>>  ZSSR run.py begin")
    print_system()
    config = CONFIG()   # 获得基准的config配置

    if config.system == "linux":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID

    # for i in range(0, 3):
    #     run_bsds100_a_image(2, config=config)
    # index=2, 30.166222309772564 0.9432615378365822

    # print(run_bsds100_a_image(2, config=config))

    run_bsds100_images()



