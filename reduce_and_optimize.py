import random
import argparse
import os.path as osp
from numpy.random import choice
import torch
import pydiffvg
import numpy as np

from tqdm import tqdm

import custom_parser
from utils import bcolors
from histogram_loss import NonDifferentiableHistogramLoss

from basic_diffvg import (VanillaDiffVG,
                          compose_image_with_white_background,
                          render_based_on_shapes_and_shape_groups)


class ReduceAndOptimize(VanillaDiffVG):
    def __init__(self, shapes_num_scheduler: list,
                 ranking_loss_type: str = 'mse',
                 ranking_l1_extent_in_convex_sum_l1_and_clip: float = 1.0,
                 ranking_clip_loss_config_file: str = 'test/config_init.npy',
                 sample_importance=True,
                 sample_beta=1,
                 *args, **kwargs):
        super(ReduceAndOptimize, self).__init__(*args, **kwargs)
        self.sample_importance = sample_importance
        self.sample_beta = sample_beta
        self.shapes_num_scheduler = shapes_num_scheduler
        self.ranking_loss_type = ranking_loss_type
        if ranking_loss_type == 'histogram':
            self.ranking_loss_fn = NonDifferentiableHistogramLoss(
                self.image_for_diffvg)
        else:
            self.ranking_loss_fn = self.get_reconstruction_loss(
                loss_type=ranking_loss_type,
                alpha=ranking_l1_extent_in_convex_sum_l1_and_clip,
                clip_loss_config_file=ranking_clip_loss_config_file)

    @staticmethod
    def create_shape_group(shape_id: int, fill_color: torch.tensor = torch.zeros(4)) -> pydiffvg.ShapeGroup:
        return pydiffvg.ShapeGroup(shape_ids=torch.tensor([shape_id]),
                                   fill_color=fill_color)

    def shapes_to_importance(self, shapes, shape_groups):
        # create dummy Bezier-curve and dummy shape group
        dummy_path = pydiffvg.Path(
            num_control_points=shapes[0].num_control_points,
            points=torch.zeros_like(shapes[0].points),
            stroke_width=torch.tensor(0.0), is_closed=True)

        shapes_importance = []
        target_image_with_white_bg = compose_image_with_white_background(self.image_for_diffvg)

        for i in tqdm(range(len(shapes))):
            if i == len(shapes) - 1:
                all_shapes_but_one = shapes[:-1]
                all_shape_groups_but_one = shape_groups[:-1]
            else:
                all_shapes_but_one = shapes[:i] + [dummy_path] + shapes[i + 1:]
                dummy_path_group = self.create_shape_group(i)
                all_shape_groups_but_one = shape_groups[:i] + [dummy_path_group] + shape_groups[i + 1:]

            image_without_one_shape = render_based_on_shapes_and_shape_groups(
                all_shapes_but_one, all_shape_groups_but_one, no_grad=True,
                canvas_width=self.canvas_width, 
                canvas_height=self.canvas_height)
            image_without_one_shape_white_bg = \
                compose_image_with_white_background(image_without_one_shape)
            # NOTE: decide if this needs to be self.image_for_diffvg OR the
            # image create from PNG2SVG_using_diffVG(target_PNG).
            loss = self.ranking_loss_fn(
                target_image_with_white_bg,
                image_without_one_shape_white_bg)
            shapes_importance.append(loss.item())
        # validate lengths
        assert len(shapes) == len(shape_groups) == len(shapes_importance)
        if self.sample_importance:
            sample_shape_importance = self.sample_according_to_importance(shapes_importance, self.sample_beta)
        else:
            sample_shape_importance = sorted(range(len(shapes_importance)), key=shapes_importance.__getitem__)
        return sample_shape_importance

    @staticmethod
    def sample_according_to_importance(shape_importance, sample_beta):
        shape_importance_tensor = torch.tensor(shape_importance[1:], dtype=torch.float32)
        shapes_weight = torch.nn.functional.softmax(shape_importance_tensor / sample_beta, dim=0)
        shapes_weight = (shapes_weight / sum(shapes_weight)).tolist()
        if sum(shapes_weight) != 1:
            diff = sum(shapes_weight) - 1
            shapes_weight[np.argmax(shapes_weight)] -= diff
        sample_shapes = choice(range(1, len(shape_importance)), len(shape_importance)-1, p=shapes_weight, replace=False)
        # Add BG at the end - so it won't be reduced
        return np.append(sample_shapes[::-1], 0)

    @staticmethod
    def carve_shapes(shapes, shape_groups, shapes_index_sample,
                     how_many_to_carve):
        shapes_subset = []
        shape_groups_subset = []
        new_shape_id = 0

        shapes_indices_to_remove = shapes_index_sample[:how_many_to_carve + 1]
        for pos, (shape, shape_group) in enumerate(zip(shapes, shape_groups)):
            if pos not in shapes_indices_to_remove:
                shapes_subset.append(shape)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([new_shape_id]),
                    fill_color=shape_group.fill_color)
                new_shape_id += 1
                shape_groups_subset.append(path_group)
        return shapes_subset, shape_groups_subset

    def run(self):
        # init:
        shapes, shape_groups = self.get_initial_shapes(self.shapes_num_scheduler[0],
                                                       self.canvas_width,
                                                       self.canvas_height, )
        self.save_intermediate_image(shapes, shape_groups, epoch=-1)
        # for loop on reduce steps:
        for epoch, (curr_num_shapes, next_num_shapes, num_iters) in enumerate(zip(
                self.shapes_num_scheduler[:-1], self.shapes_num_scheduler[1:], self.num_iterations[:-1])):
            # optimize:
            shapes, shape_groups = self.optimize_shapes(
                self.image_for_diffvg, shapes, shape_groups, num_iters, epoch)
            self.save_intermediate_image(shapes, shape_groups, 2 * epoch)
            self.save_svg_image_by_name(shapes, shape_groups,
                                        f"after_optimization_{len(shapes):04d}")
            assert len(shapes) == curr_num_shapes
            # rank:
            shapes_rank = self.shapes_to_importance(shapes, shape_groups)
            assert len(shapes) == curr_num_shapes
            # reduce:
            how_many_to_carve = curr_num_shapes - next_num_shapes - 1
            shapes, shape_groups = self.carve_shapes(shapes, shape_groups,
                                                     shapes_rank,
                                                     how_many_to_carve)
            assert len(shapes) == next_num_shapes
            self.save_intermediate_image(shapes, shape_groups, 2 * epoch + 1)

        # lastly, we need to optimize:
        total_num_epochs = 2 * len(self.shapes_num_scheduler)
        shapes, shape_groups = self.optimize_shapes(
            self.image_for_diffvg, shapes, shape_groups, self.num_iterations[-1],
            2 * total_num_epochs)
        self.save_final_result(shapes, shape_groups)


def main(script_args):
    image_name = script_args.target.split("/")[-1].split(".")[0]
    scheduler = [int(x) for x in script_args.scheduler]
    iterations = [int(x) for x in script_args.num_iter]
    num_paths = scheduler[0]
    recons_loss_type = script_args.recons_loss_type
    geometric_loss_type = script_args.geometric_loss_type
    if script_args.experiment_name != '':
        experiment_name = script_args.experiment_name
    else:
        experiment_name = f'reduce_and_optimize_' \
                          f'{image_name}_{num_paths}_' \
                          f'rec_{recons_loss_type}_' \
                          f'geom_{geometric_loss_type}'
        if script_args.advanced_logging:
            experiment_name += "_advanced_logging"

    root_out_dir = osp.join(script_args.results_dir, experiment_name)

    diffvg_runner = ReduceAndOptimize(
        shapes_num_scheduler=scheduler,
        ranking_loss_type=script_args.ranking_loss_type,
        ranking_l1_extent_in_convex_sum_l1_and_clip=script_args.ranking_l1_and_clip_alpha,
        ranking_clip_loss_config_file=script_args.ranking_clip_config_file,
        path_to_png_image=script_args.target,
        num_paths=num_paths,
        num_iterations=iterations,
        epochs=1,
        root_output_directory=root_out_dir,
        canvas_height=script_args.canvas_height,
        canvas_width=script_args.canvas_width,
        reconstruction_loss_type=recons_loss_type,
        l1_extent_in_convex_sum_l1_and_clip=script_args.l1_and_clip_alpha,
        lambda_geometric=script_args.lambda_geometric,
        geometric_loss_lamda_geometric_punish=script_args.geometric_loss_lamda_geometric_punish,
        clip_config_file=script_args.clip_config_file,
        geometric_loss_type=geometric_loss_type,
        is_advanced_logging=script_args.advanced_logging,
        sample_importance=script_args.sample_importance,
        text_prompt=script_args.text_prompt,
        init_type=script_args.init_type,
        init_shape=script_args.init_shape,
    )
    diffvg_runner.run()


if __name__ == "__main__":
    args = parser.parse_arguments()
    print(args)
    main(args)
