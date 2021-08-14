from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import DataOp
import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
from Config import CONFIG


def evaluate_bsds100_a_image(dataset_path: str, index: int, method: str):
    sr = DataOp.read_BSDS100(dataset_path=dataset_path, index=index, type=method)
    hr = DataOp.read_BSDS100(dataset_path=dataset_path, index=index, type="HR")

    sr = F.to_tensor(sr).numpy()
    hr = F.to_tensor(hr).numpy()

    sr = np.transpose(sr, (1, 2, 0))
    hr = np.transpose(hr, (1, 2, 0))

    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(hr, sr)
    ssim = structural_similarity(hr, sr, multichannel=True)

    print(psnr, ssim)
    return psnr, ssim



if __name__ == '__main__':
    config = CONFIG()
    for i in range(1, 101):
        psnr, ssim = evaluate_bsds100_a_image(config.BSDS100xN_PATH[2], 2, "SRCNN")





# if __name__ == '__main__':

