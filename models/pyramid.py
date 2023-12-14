import torch
import numpy as np
import torch.nn.functional as F

def gaussian_kernel(size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    interval  = ( 2 *sigma +1) / (size)
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
    kernel_tensor.to(device)
    return kernel_tensor


def gaussian_conv2d(x, g_kernel, dtype=torch.float):
    # Assumes input of x is of shape: (minibatch, depth, height, width)
    # Infer depth automatically based on the shape
    channels = g_kernel.shape[0]
    padding = g_kernel.shape[-1] // 2  # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
    return y


def downsample(x):
    # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    return x[:, :, ::2, ::2]


def create_laplacian_pyramid(x, kernel, levels):
    # upsample = torch.nn.Upsample(scale_factor=2)  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    current_x = x
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        # laplacian = current_x - upsample(down)
        pyramids.append(down)
        current_x = down
    pyramids.append(current_x)
    return pyramids


class MSEPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=4, channels=3, kernel_size=5, sigma=1, device=torch.device('cpu'), dtype=torch.float):
        super(MSEPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype).to(device)

    def forward(self, x, target):
        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = create_laplacian_pyramid(target, self.kernel, self.max_levels)
        return sum(torch.nn.functional.mse_loss(x, y) for x, y in zip(input_pyramid, target_pyramid))
