import platform


class CONFIG:
    def __init__(self):

        # 各类路径的配置
        # Linux:
        if platform.system().lower() == "linux":
            self.system = "linux"
            self.PROJECT_PATH = "/home/gaoruopeng/code/ZSSR"
            self.BSDS100_PATH = "/data0/gaoruopeng/BSDS100"     # BSDS100 数据集路径
            self.BSDS100xN_PATH = [
                None,
                None,
                "/data0/gaoruopeng/BSDS100/image_SRF_2",
                "/data0/gaoruopeng/BSDS100/image_SRF_3",
                "/data0/gaoruopeng/BSDS100/image_SRF_4",
            ]
        else:
            self.system = "windows"
            self.PROJECT_PATH = "D:/ResearchFile/Code/ZSSR"
            self.BSDS100_PATH = "D:/ResearchFile/Dataset/BSDS100"  # BSDS100 数据集路径
            self.BSDS100xN_PATH = [
                None,
                None,
                "D:/ResearchFile/Dataset/BSDS100/image_SRF_2",
                "D:/ResearchFile/Dataset/BSDS100/image_SRF_3",
                "D:/ResearchFile/Dataset/BSDS100/image_SRF_4",
            ]

        self.DATASET = "BSDS100"     # 选择运行时的数据集
        self.SCALE_FACTOR = 2    # 超分辨率因子，一般是整数

        # Train部分的参数
        self.LEARN_RATE = 0.005
        self.WEIGHT_DECAY = 0.0

        self.KERNEL_SIZE = 3
        self.PADDING = 1
        self.PADDING_MODE = "zero"

        self.INIT_EPOCH = 0
        self.FINAL_EPOCH = None
        self.MIN_LR = 0.00001  # 最低学习率，如果当前的学习率低于这个值，则停止学习

        self.NET_IMAGE_SIZE = 128   # 网络最终输入的图像的大小

        self.LOSS_NEIGHBOR_LEN = 50
        self.LR_DROP_WHEN = 1.5  # 如果标准差大于斜率*系数，则LR减少
        self.LR_DROP_RATE = 10  # 学习率降低因子

        # 调试信息
        self.PRINT_TRAIN_EPOCH = True    # 每一轮训练的输出

        # 硬件信息
        self.DEVICE = "cuda"



