import platform


class Config:
    def __init__(self):

        # 各类路径的配置
        # Linux:
        if platform.system().lower() == "linux":
            PROJECT_PATH = "/home/gaoruopeng/code/ZSSR/"
            BSDS100_PATH = "/data0/gaoruopeng/BSDS100/"     # BSDS100 数据集路径
            BSDS100xN_PATH = [
                None,
                None,
                "/data0/gaoruopeng/BSDS100/image_SRF_2/",
                "/data0/gaoruopeng/BSDS100/image_SRF_3/",
                "/data0/gaoruopeng/BSDS100/image_SRF_4/",
            ]
        else:
            PROJECT_PATH = "D:\ResearchFile\Code\ZSSR"
            BSDS100_PATH = "D:\ResearchFile\Dataset\BSDS100"  # BSDS100 数据集路径
            BSDS100xN_PATH = [
                None,
                None,
                "D:\ResearchFile\Dataset\BSDS100/image_SRF_2/",
                "D:\ResearchFile\Dataset\BSDS100/image_SRF_3/",
                "D:\ResearchFile\Dataset\BSDS100/image_SRF_4/",
            ]

        DATASET = "BSDS100"     # 选择运行时的数据集
        SCALE_FACTOR = 2    # 超分辨率因子，一般是整数


