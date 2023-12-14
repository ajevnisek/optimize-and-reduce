import os

import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def save_grid_to_path(imgs, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path)


def create_fig_from_images(root_dir_images, path_video):
    filenames = [os.path.join(root_dir_images, x) for x in os.listdir(root_dir_images) if x.endswith('.png')]
    filenames.sort()
    import imageio
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(path_video, images)

