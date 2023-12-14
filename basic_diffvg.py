import os
import json
import yaml
import time
import argparse
import os.path as osp
import cv2
import webp
import numpy as np
import pydiffvg
import torch
import skimage
import skimage.io
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import skimage.metrics as metrics
import pickle
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datetime import datetime
from functools import partial
from cairosvg import svg2png
from torchvision.utils import make_grid
from dataclasses import dataclass
from geometric_loss import GeometryLoss
# run: pip install kornia==0.5.0
# rurn: pip install webp
import kornia

from utils import bcolors
from custom_parser import parse_arguments

dsample = kornia.transform.PyrDown()


def add_to_file(data_to_add, timing_file):
    data = {}
    # read all data that you have so far:
    if os.path.exists(timing_file):
        with open(timing_file, 'r') as f:
            data = json.load(f)
    # update dict:
    for k in data_to_add:
        data[k] = data_to_add[k]
    # write dict to file:
    with open(timing_file, 'w') as f:
        json.dump(data, f, indent=2)


def gaussian_pyramid_loss(loss_fn, recons, input):
    recons = recons.permute(2, 0, 1).unsqueeze(0)
    input = input.permute(2, 0, 1).unsqueeze(0)
    recon_loss = loss_fn(recons, input, reduction='none').mean(
        dim=[1, 2, 3])  # + self.lpips(recons, input)*0.1
    for j in range(2, 5):
        recons = dsample(recons)
        input = dsample(input)
        recon_loss = recon_loss + loss_fn(recons, input,
                                          reduction='none').mean(
            dim=[1, 2, 3]) / j
    return recon_loss


gamma = 1.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_rgba(tensor):
    return tensor.shape[-1] == 4


def render_based_on_shapes_and_shape_groups(shapes, shape_groups,
                                            no_grad=True, canvas_width=256,
                                            canvas_height=256, ):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width=canvas_width, canvas_height=canvas_height, shapes=shapes, shape_groups=shape_groups)
    if no_grad:
        with torch.no_grad():
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,  # width
                         canvas_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,
                         *scene_args)
    else:
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width,  # width
                     canvas_height,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     0,  # seed
                     None,
                     *scene_args)
    return img


def compose_image_with_white_background(img: torch.tensor) -> torch.tensor:
    if img.shape[-1] == 3:  # return img if it is already rgb
        return img
    # Compose img with white background
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + (1 - alpha) * torch.ones(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device())
    return img


def compose_image_with_black_background(img: torch.tensor) -> torch.tensor:
    if img.shape[-1] == 3:  # return img if it is already rgb
        return img
    # Compose img with white background
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + (1 - alpha) * torch.zeros(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device())
    return img


def init_clip_loss(clip_loss_config="test/config_init.npy", text_prompt=None):
    from models.loss import CLIPLoss, CLIPConvLoss
    cfg = argparse.Namespace(**np.load(clip_loss_config, allow_pickle=True).item())
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_loss = CLIPLoss(cfg, text_prompt=text_prompt)
    clip_weight = cfg.clip_weight
    clip_conv_loss = CLIPConvLoss(cfg)
    return clip_loss, clip_weight, clip_conv_loss


class PNG2SVG:
    """This module implements an algorithm to convert PNG images to SVG.

    The run module determines the concrete algorithm.
    """

    def __init__(self, path_to_png_image, text_prompt=None, *args, **kwargs):
        self.text_prompt = text_prompt
        self.path_to_png_image = path_to_png_image

    @staticmethod
    def eval_clip_loss(clip_loss, clip_weight, clip_conv_loss,
                       rendered_image: torch.tensor,
                       target_image: torch.tensor, ):
        if is_rgba(rendered_image):
            rendered_image = compose_image_with_white_background(rendered_image)
        if is_rgba(target_image):
            target_image = compose_image_with_white_background(target_image)
        h, w, c = target_image.shape
        longest_edge = max(h, w)
        target_reshaped = F.interpolate(
            target_image.unsqueeze(0).unsqueeze(0),
            size=(longest_edge, longest_edge, c)).squeeze(0).squeeze(0)
        rendered_reshaped = F.interpolate(
            rendered_image.unsqueeze(0).unsqueeze(0),
            size=(longest_edge, longest_edge, c)).squeeze(0).squeeze(0)
        reconstruction_loss = sum(
            clip_loss(rendered_reshaped.permute(2, 0, 1).unsqueeze(0),
                      target_reshaped.permute(2, 0, 1).unsqueeze(0))
        ) * clip_weight + sum(
            list(
                clip_conv_loss(
                    rendered_reshaped.permute(2, 0, 1).unsqueeze(0),
                    target_reshaped.permute(2, 0, 1).unsqueeze(0)
                ).values()
            )
        )
        return reconstruction_loss

    def get_reconstruction_loss(self, loss_type, alpha=1.0,
                                clip_loss_config_file="test/config_init.npy", ):
        clip_loss, clip_weight, clip_conv_loss = init_clip_loss(
            clip_loss_config_file, text_prompt=self.text_prompt)

        def l1_and_clip_mix(l1_extent, target_image, reconstructed_image):
            return l1_extent * torch.nn.L1Loss()(
                target_image, reconstructed_image) + (1.0 - l1_extent) * \
                self.eval_clip_loss(clip_loss, clip_weight,
                                    clip_conv_loss, target_image,
                                    reconstructed_image)

        losses = {
            'mse': torch.nn.MSELoss(),
            'l1': torch.nn.L1Loss(),
            'pyramid-mse': partial(gaussian_pyramid_loss, torch.nn.functional.mse_loss, ),
            'pyramid-l1': partial(gaussian_pyramid_loss, torch.nn.functional.l1_loss, ),
            'clip': partial(self.eval_clip_loss, clip_loss, clip_weight,
                            clip_conv_loss),
            'l1_and_clip_mix': partial(l1_and_clip_mix, alpha)
        }
        return losses[loss_type]

    def get_geometric_loss(self, loss_type: str, lamda_geometric_punish: float = 10.0, *args, **kwargs):
        losses = {
            'none': lambda _: torch.tensor([0]).to(device),
            'geometric': GeometryLoss(lamda_geometric_punish=lamda_geometric_punish)
        }
        return losses[loss_type]

    def run(self):
        pass


@dataclass
class RunParameters:
    experiment_name: str
    root_output_directory: str
    intermediate_results_directory: str


class VanillaDiffVG(PNG2SVG):
    def __init__(self, path_to_png_image: str, num_paths: int,
                 num_iterations: list, epochs: int, root_output_directory: str,
                 canvas_width: int, canvas_height: int,
                 reconstruction_loss_type: str,
                 geometric_loss_type: str,
                 is_advanced_logging: bool = False,
                 l1_extent_in_convex_sum_l1_and_clip: float = 1.0,
                 geometric_loss_lamda_geometric_punish: float = 10.0,
                 lambda_geometric=0.1,
                 clip_loss_config_file: str = "test/config_init.npy",
                 text_prompt=None,
                 init_type='custom',
                 init_shape='circle',
                 timing_json_path: str = '',
                 final_loss_path: str = '',
                 early_stopping: bool = False,
                 input_svg=None,
                 *args, **kwargs):
        """

        """
        super().__init__(path_to_png_image, text_prompt, args, kwargs)
        # initialize gpu related variables
        self.device = device
        self.initialize_diffvg()

        # load the image and adjust canvas:
        self.raw_tensor_image = self.read_png_image_from_path(path_to_png_image)
        h, w = self.raw_tensor_image.shape[0], self.raw_tensor_image.shape[1]
        self.canvas_height = canvas_height if canvas_height > 0 else h
        self.canvas_width = canvas_width if canvas_width > 0 else w
        self.image_for_diffvg = self.prepare_png_image_for_diffvg(
            self.raw_tensor_image, self.canvas_width, self.canvas_height)

        # initialize svg and optimization parameters:
        self.num_paths = num_paths
        self.num_iterations = num_iterations
        self.epochs = epochs
        self.init_type = init_type
        self.init_shape = init_shape
        self.early_stopping = early_stopping
        self.final_loss = 0

        # initialize losses:
        self.geometric_loss_type = geometric_loss_type
        self.geometric_loss_fn = self.get_geometric_loss(loss_type=geometric_loss_type,
                                                         lamda_geometric_punish=geometric_loss_lamda_geometric_punish)
        self.lambda_geometric = lambda_geometric
        self.reconstruction_loss_type = reconstruction_loss_type
        self.reconstruction_loss_fn = self.get_reconstruction_loss(
            loss_type=reconstruction_loss_type,
            alpha=l1_extent_in_convex_sum_l1_and_clip,
            clip_loss_config_file=clip_loss_config_file)

        # initialize logging and run parameters:
        self.run_parameters = self.initialize_run_parameters(
            path_to_png_image, root_output_directory)
        self.timing_file = timing_json_path
        self.final_loss_file = final_loss_path
        now = datetime.now()
        self.tb_logger = SummaryWriter(
            self.run_parameters.root_output_directory,
            comment=f"{now.strftime('%Y_%m_%d_%H_%M_%S')}")
        self.is_advanced_logging = is_advanced_logging
        self.input_svg = input_svg
        # init seed
        self.init_seed()

    @staticmethod
    def init_seed(seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def initialize_diffvg() -> None:
        pydiffvg.set_use_gpu(torch.cuda.is_available())

    @staticmethod
    def initialize_run_parameters(path_to_png_image: str,
                                  root_output_directory: str) -> RunParameters:
        # convert the full image path to a simple name:
        experiment_name = path_to_png_image.split("/")[-1].split(".")[0]
        # create directories:
        intermediate_results_directory = osp.join(
            root_output_directory, 'intermediate_results')
        os.makedirs(root_output_directory, exist_ok=True)
        os.makedirs(intermediate_results_directory, exist_ok=True)
        # return script parameters:
        return RunParameters(experiment_name, root_output_directory,
                             intermediate_results_directory)

    @staticmethod
    def read_png_image_from_path(path_to_png_image: str) -> torch.tensor:
        if path_to_png_image.endswith('.webp'):
            numpy_image = np.array(webp.load_image(path_to_png_image))
        elif path_to_png_image.endswith('.svg'):
            def svg_code_to_pil_image(svg_code, format='RGBA'):
                fg = Image.open(BytesIO(svg2png(bytestring=svg_code))).convert(
                    format)
                new_image = Image.new("RGBA", fg.size,
                                      "WHITE")  # Create a white rgba background
                new_image.paste(fg, (0, 0), fg)
                return new_image.convert('RGB')

            svg_code = ''
            with open(path_to_png_image, 'r') as f:
                svg_code = f.read()
            numpy_image = np.array(svg_code_to_pil_image(svg_code))
        else:
            numpy_image = skimage.io.imread(path_to_png_image)
        normalized_tensor_image = torch.from_numpy(numpy_image).to(
            torch.float32) / 255.0
        return normalized_tensor_image

    @staticmethod
    def prepare_png_image_for_diffvg(normalized_tensor_image: torch.tensor,
                                     canvas_width: int, canvas_height: int) -> \
            torch.tensor:
        # load the image to the same device diffvg is on:
        normalized_tensor_image = normalized_tensor_image.to(
            pydiffvg.get_device())
        h, w = normalized_tensor_image.shape[0], \
            normalized_tensor_image.shape[1]
        # resize the image according to diffvg dimensions:
        canvas_height = canvas_height if canvas_height > 0 else h
        canvas_width = canvas_width if canvas_width > 0 else w
        resizer = torchvision.transforms.Resize((canvas_height, canvas_width))
        resized_image = resizer(normalized_tensor_image.permute(2, 0, 1)
                                ).permute(1, 2, 0)
        return resized_image

    def log_scalars_to_tensorboard(self, name_to_loss_and_num_iter: dict) -> \
            None:
        for name in name_to_loss_and_num_iter:
            loss, iteration_number = name_to_loss_and_num_iter[name]
            self.tb_logger.add_scalar(name, loss, iteration_number)

    def log_img_to_tensorboard(self, iteration_number, shapes, shape_groups):
        img = render_based_on_shapes_and_shape_groups(shapes, shape_groups)
        self.save_svg_image_from_shapes(shapes, shape_groups, osp.join(
            self.run_parameters.root_output_directory, 'intermediate_results', f'iter_{iteration_number}.svg'))
        self.tb_logger.add_image("Intermediate Results", img.permute(2, 0, 1), iteration_number)

    @staticmethod
    def measure_loss_from_tensor_to_tensor(target_image, reconstructed_image,
                                           loss_function):
        try:
            # handles the case of psnr and rgb / rgba images
            if is_rgba(target_image):
                target_image = compose_image_with_white_background(target_image)
                loss = loss_function(
                    compose_image_with_white_background(target_image).cpu(

                    ).numpy() if is_rgba(target_image) else
                    target_image.cpu().numpy(),
                    compose_image_with_white_background(reconstructed_image).cpu(

                    ).numpy() if is_rgba(reconstructed_image) else
                    reconstructed_image.cpu().numpy())
            else:
                loss = loss_function(target_image.cpu().numpy(),
                                     reconstructed_image.cpu().numpy())
        except:
            loss = loss_function(target_image.cpu(),
                                 reconstructed_image.cpu()).item()
        return loss

    def measure_svg_to_tensor_image_loss(self, shapes, shape_groups,
                                         tensor_image,
                                         loss_fn, canvas_width=256,
                                         canvas_height=256):
        svg_img_in_tensor = render_based_on_shapes_and_shape_groups(
            shapes, shape_groups, no_grad=True, canvas_width=canvas_width,
            canvas_height=canvas_height, )
        svg_img_in_tensor = compose_image_with_white_background(
            svg_img_in_tensor)
        return self.measure_loss_from_tensor_to_tensor(
            target_image=tensor_image, reconstructed_image=svg_img_in_tensor,
            loss_function=loss_fn)

    def report_advanced_stats_to_logger(self, shapes, shape_groups,
                                        resized_target, iteration_number):
        self.tb_logger.add_scalar('num_shapes', len(shapes), iteration_number)
        self.tb_logger.add_scalar('PSNR',
                                  self.measure_svg_to_tensor_image_loss(
                                      shapes, shape_groups,
                                      tensor_image=resized_target,
                                      loss_fn=metrics.peak_signal_noise_ratio,
                                      canvas_width=self.canvas_width,
                                      canvas_height=self.canvas_height,
                                  ),
                                  iteration_number)
        self.tb_logger.add_scalar('MSE',
                                  self.measure_svg_to_tensor_image_loss(
                                      shapes, shape_groups,
                                      canvas_width=self.canvas_width,
                                      canvas_height=self.canvas_height,
                                      tensor_image=resized_target,
                                      loss_fn=torch.nn.MSELoss()),
                                  iteration_number)
        self.tb_logger.add_scalar('L1',
                                  self.measure_svg_to_tensor_image_loss(
                                      shapes, shape_groups,
                                      canvas_width=self.canvas_width,
                                      canvas_height=self.canvas_height,
                                      tensor_image=resized_target,
                                      loss_fn=torch.nn.L1Loss()),
                                  iteration_number)

    @staticmethod
    def sample_init_points(img: torch.tensor,
                           curr_img: torch.tensor,
                           num_points: int):
        resized_tensor = torch.nn.functional.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            (100, 100)).squeeze(0).permute(1, 2, 0)
        if curr_img is None:
            resized_curr_img = torch.zeros_like(resized_tensor)
        else:
            resized_curr_img = torch.nn.functional.interpolate(
                curr_img.permute(2, 0, 1).unsqueeze(0),
                (100, 100)).squeeze(0).permute(1, 2, 0)
        numpy_image = np.abs((resized_tensor-resized_curr_img).cpu().numpy()) * 255.0
        h, w, c = numpy_image.shape
        X = numpy_image[..., :3].reshape(h * w, -1)
        clustering = DBSCAN(eps=5, min_samples=20).fit(X)  # good

        labels = clustering.labels_
        labels_image = labels.reshape(h, w)
        sorted_labels = np.unique(labels_image)
        sorted_labels.sort()

        centers = []
        for l in sorted_labels[1:]:
            totalLabels, label_ids, values, centroid = \
                cv2.connectedComponentsWithStats((labels_image == l).astype(np.uint8), connectivity=4)
            # Loop through each component
            for i in range(1, totalLabels):
                # Area of the component
                area = values[i, cv2.CC_STAT_AREA]
                center = centroid[i]
                center = np.stack((center[0] * (img.shape[0] / 100.), center[1] * (img.shape[1] / 100.)))
                color = torch.cat((resized_tensor[labels_image == l].median(dim=0)[0],
                                   torch.ones(1, device=resized_tensor.device)))
                centers.append((center, color, np.sqrt(area / np.pi)))

        sorted_components = sorted(centers, key=lambda x: x[-1], reverse=True)
        return [x[0] for x in sorted_components][:num_points], \
            [x[1] for x in sorted_components][:num_points], \
            [x[2] for x in sorted_components]

    @staticmethod
    def get_bg_color(img: torch.tensor, ):
        h, w, c = img.shape
        img = compose_image_with_white_background(img)
        edges = torch.cat((img[0], img[:, 0], img[h - 1],
                           img[:, w - 1])).reshape(-1, 3)
        median = torch.cat((edges.median(dim=0)[0],
                            torch.tensor([1.], device=edges.device)))
        return median

    @staticmethod
    def get_bezier_circle(radius: float = 1,
                          segments: int = 4,
                          bias: np.array = None):
        if bias is None:
            bias = (random.random(), random.random())
        deg = torch.arange(0, segments * 3 + 1) * 2 * np.pi / (segments * 3 + 1)
        points = torch.stack((torch.cos(deg), torch.sin(deg))).T
        points = points * radius + torch.tensor(bias).unsqueeze(dim=0)
        points = points.type(torch.FloatTensor).contiguous()
        return points

    def add_bg(self, target_img: torch.tensor,
               canvas_width: int = 256,
               canvas_height: int = 256, ):
        bg_color = self.get_bg_color(target_img)
        bg_pts = torch.tensor([[-10, -10],
                               [-9, -10],
                               [canvas_width, 0],
                               [canvas_width + 1, 0],
                               [canvas_width + 2, 0],
                               [canvas_width, canvas_height],
                               [canvas_width + 1, canvas_height],
                               [canvas_width + 1, canvas_height + 1],
                               [0, canvas_height],
                               [0, canvas_height + 1],
                               [0, canvas_height],
                               [0, canvas_height + 1],
                               [-10, -10]],
                              dtype=torch.float32)
        if target_img.shape[-1] == 3:
            bg_img = torch.ones_like(target_img) * bg_color[:3]
        else:
            bg_img = torch.ones_like(target_img) * bg_color
        return bg_pts, bg_color, bg_img

    def get_initial_shapes(self,
                           num_paths: int,
                           canvas_width: int = 256,
                           canvas_height: int = 256,
                           current_pic: torch.tensor = None):
        shapes = []
        shape_groups = []
        num_segments = 4
        num_control_points = torch.tensor(num_segments * [2] + [0],
                                          dtype=torch.int32)
        target_image_with_white_background = compose_image_with_white_background(self.image_for_diffvg)
        if current_pic is None:
            # ADD Background
            bg_pts, bg_color, _ = self.add_bg(target_image_with_white_background, canvas_width, canvas_height)
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=bg_pts,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=bg_color)
            shape_groups.append(path_group)
            num_paths -= 1
        init_points, init_colors, radiuses = self.sample_init_points(target_image_with_white_background,
                                                                     current_pic,
                                                                     num_paths)
        for i in range(num_paths):
            if i > len(init_points)-1 or self.init_type == 'random':
                p0 = (random.random() * canvas_width, random.random() * canvas_height)
                color = torch.tensor([random.random(), random.random(),
                                      random.random(), random.random() * 0.5])
                radius = 5e-3 * canvas_width
            else:
                p0 = init_points[i]
                color = init_colors[i]
                radius = radiuses[i]
            if self.init_shape == "random":
                points = []
                points.append(p0)
                for j in range(num_segments):
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] +
                          radius * (random.random() - 0.5))
                    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] +
                          radius * (random.random() - 0.5))
                    points.append(p1)
                    points.append(p2)
                    if j < num_segments - 1:
                        p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] +
                              radius * (random.random() - 0.5))
                    else:
                        p0 = p3
                    points.append(p3)
                points = torch.tensor(points)
            else:
                points = self.get_bezier_circle(radius=radius, segments=num_segments, bias=p0)

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color)
            shape_groups.append(path_group)
        return shapes, shape_groups

    def optimize_shapes(self, target_image: torch.tensor, shapes, shape_groups,
                        num_iter: int, epoch: int = 0, early_stopping: bool = False):
        points_vars = []
        color_vars = []

        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)

        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        points_optim = torch.optim.Adam(points_vars, lr=1.)
        color_optim = torch.optim.Adam(color_vars, lr=1e-2)
        rendered_image = torch.zeros_like(target_image)

        # Optimize
        pbar = tqdm(range(num_iter))
        loss_list = []
        for t in pbar:
            points_optim.zero_grad()
            color_optim.zero_grad()

            # render image with current shapes:
            rendered_image = render_based_on_shapes_and_shape_groups(
                shapes, shape_groups, no_grad=False,
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height)
            if rendered_image.shape[-1] == 4:
                rendered_image = compose_image_with_white_background(
                    rendered_image)
            if target_image.shape[-1] == 4:
                target_image = compose_image_with_white_background(target_image)
            # calculate losses:
            geometric_loss = self.geometric_loss_fn(shapes)
            reconstruction_loss = self.reconstruction_loss_fn(
                rendered_image, target_image)

            loss = reconstruction_loss + self.lambda_geometric * geometric_loss
            loss_list.append(loss.item())
            # Back-propagate the gradients:
            loss.backward()

            # Take a gradient descent step:
            points_optim.step()
            color_optim.step()
            # clamp colors to [0, 1]:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            # log losses to tensorboard and shell
            iteration_number = (epoch * num_iter) + t
            name_to_loss_and_iteration_number = {
                f'Reconstruction Loss {self.reconstruction_loss_type}': (
                    reconstruction_loss.item(), iteration_number),
                f'Geometric Loss {self.geometric_loss_type}': (
                    geometric_loss.item(), iteration_number),
                'Loss': (loss.item(), iteration_number)
            }
            self.log_scalars_to_tensorboard(name_to_loss_and_iteration_number)
            # if iteration_number % 5 == 0:
            #     self.log_img_to_tensorboard(iteration_number, shapes, shape_groups)
            pbar.set_description(' | '.join([
                f"{name}: {v[0]:.2f}" for name, v in
                name_to_loss_and_iteration_number.items()
            ]))
            if self.is_advanced_logging:
                self.report_advanced_stats_to_logger(
                    shapes, shape_groups, target_image, iteration_number)
            # flush:
            torch.cuda.empty_cache()
            if len(loss_list) > 50 and early_stopping:
                loss_improvement = (1 - loss_list[-1] / loss_list[-2]) * 100.0
                if loss_improvement <= 1e-2:
                    break
        self.tb_logger.add_image('Image', rendered_image.permute(2, 0, 1),
                                 epoch)
        self.final_loss = loss.item()
        add_to_file({'floss_list': loss_list}, osp.join(osp.dirname(self.final_loss_file),
                                                        f'loss_epoch_{epoch:02d}.json'))

        return shapes, shape_groups

    def render_tensor_image_from_shapes(self, shapes, shape_groups):
        gt_image = render_based_on_shapes_and_shape_groups(
            shapes, shape_groups, canvas_width=self.canvas_width,
            canvas_height=self.canvas_height)
        gt_image_white_bg = compose_image_with_white_background(gt_image)
        return gt_image_white_bg

    @staticmethod
    def save_tensor_image_to_path(tensor_image: torch.tensor, path: str):
        pydiffvg.imwrite(tensor_image.cpu(), path, gamma=1.)

    def save_svg_image_from_shapes(self, shapes, shape_groups, path):
        pydiffvg.save_svg(path, self.canvas_width, self.canvas_height,
                          shapes, shape_groups)

    def save_intermediate_image(self, shapes, shape_groups, epoch):
        tensor_img = self.render_tensor_image_from_shapes(shapes, shape_groups)
        path_to_generated_image = osp.join(
            self.run_parameters.intermediate_results_directory,
            f'rasterized_image_epoch_{epoch:02d}.png')
        self.save_tensor_image_to_path(
            tensor_img, path_to_generated_image)
        path_to_svg_image = osp.join(
            self.run_parameters.intermediate_results_directory,
            f'svg_image_epoch_{epoch:02d}.svg')
        self.save_svg_image_from_shapes(shapes, shape_groups, path_to_svg_image)

    def save_svg_image_by_name(self, shapes, shape_groups, name):
        tensor_img = self.render_tensor_image_from_shapes(shapes, shape_groups)
        path_to_generated_image = osp.join(
            self.run_parameters.intermediate_results_directory,
            name + '.png')
        self.save_tensor_image_to_path(
            tensor_img, path_to_generated_image)
        self.save_svg_image_from_shapes(shapes, shape_groups, osp.join(
            self.run_parameters.intermediate_results_directory, name + '.svg'))

    def save_final_result(self, shapes, shape_groups):
        tensor_img = self.render_tensor_image_from_shapes(shapes, shape_groups)
        path_to_generated_image = osp.join(
            self.run_parameters.root_output_directory,
            f'result.png')
        self.save_tensor_image_to_path(
            tensor_img, path_to_generated_image)
        self.save_svg_image_from_shapes(shapes, shape_groups, osp.join(
            self.run_parameters.root_output_directory, 'result.svg'))

    def run(self):
        add_to_file({'1_start': time.time()}, self.timing_file)
        shapes, shape_groups = self.get_initial_shapes(self.num_paths,
                                                       self.canvas_width,
                                                       self.canvas_height, )
        for epoch in range(self.epochs):
            shapes, shape_groups = self.optimize_shapes(
                self.image_for_diffvg, shapes, shape_groups,
                int(self.num_iterations[epoch]), epoch, early_stopping=self.early_stopping)
            self.save_intermediate_image(shapes, shape_groups, epoch)
        add_to_file({'2_end': time.time()}, self.timing_file)
        add_to_file({'final_loss': self.final_loss}, self.final_loss_file)
        self.save_final_result(shapes, shape_groups)


def main(script_args):
    image_name = script_args.target.split("/")[-1].split(".")[0]
    num_paths = script_args.num_paths
    recons_loss_type = script_args.recons_loss_type
    geometric_loss_type = script_args.geometric_loss_type
    if script_args.experiment_name != '':
        experiment_name = script_args.experiment_name
    else:
        experiment_name = f'{image_name}_{num_paths}_rec_{recons_loss_type}_' \
                          f'geom_{geometric_loss_type}'

    root_out_dir = osp.join(script_args.results_dir, experiment_name)
    # debug: uncomment this if you want to run diffvg only for stuff which
    # did not finish:
    # if os.path.exists(os.path.join(root_out_dir, 'result.png')):
    #     return
    print(f"{bcolors.OKGREEN}running experiment {experiment_name}... {bcolors.ENDC}")

    print(f'{bcolors.OKCYAN}experiment args: {yaml.dump(vars(script_args))} {bcolors.ENDC}')
    config_dir = os.path.join(root_out_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        f.write(yaml.dump(vars(script_args)))

    timing_dir = os.path.join(root_out_dir, 'timing')
    os.makedirs(timing_dir, exist_ok=True)
    timing_file = os.path.join(timing_dir, 'timing.json')
    loss_dir = os.path.join(root_out_dir, 'loss')
    os.makedirs(loss_dir, exist_ok=True)
    final_loss_path = os.path.join(loss_dir, 'loss.json')

    add_to_file({'0_total_start': time.time()}, timing_file)
    diffvg_runner = VanillaDiffVG(
        path_to_png_image=script_args.target,
        num_paths=num_paths,
        num_iterations=script_args.num_iter,
        epochs=1,
        root_output_directory=root_out_dir,
        canvas_height=script_args.canvas_height,
        canvas_width=script_args.canvas_width,
        reconstruction_loss_type=recons_loss_type,
        geometric_loss_type=geometric_loss_type,
        is_advanced_logging=script_args.advanced_logging,
        l1_extent_in_convex_sum_l1_and_clip=script_args.l1_and_clip_alpha,
        lambda_geometric=script_args.lambda_geometric,
        geometric_loss_lamda_geometric_punish=script_args.geometric_loss_lamda_geometric_punish,
        clip_config_file=script_args.clip_config_file,
        init_type=script_args.init_type,
        init_shape=script_args.init_shape,
        timing_json_path=timing_file,
        final_loss_path=final_loss_path,
        early_stopping=script_args.early_stopping,
    )
    diffvg_runner.run()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
