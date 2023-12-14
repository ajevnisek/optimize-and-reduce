import os.path as osp
import random

import numpy as np
import torch
import pydiffvg
from basic_diffvg import compose_image_with_white_background, add_to_file
from custom_parser import parse_arguments
from reduce_and_optimize import ReduceAndOptimize
from utils import bcolors
import yaml
import os
import time


class ReduceOrAddAndOptimize(ReduceAndOptimize):
    def __init__(self, *args, **kwargs):
        super(ReduceOrAddAndOptimize, self).__init__(*args, **kwargs)

    @staticmethod
    def interleave_lists(list1, indices, list3):
        """Interleave lists and get new indices.

        Interleave the elements of list3 into list1 at the positions
        indicated by indices. Returns the interleaved list and an index map
        from the original positions of list1 to the new positions of the
        returned list.

        Args:
            list1: list. The base list of items.
            indices: list. The indices in which you want to interleave list3
            items in list1.
            list3: list. The list of items to put in the indices of list1.
        Returns: tuple. The first item is the interleaved list.
                 The second item is a map from old indices of list1 to the new
                 indices of list1.
                 The third item is a map from old indices of list3 to the new
                 indices of list3.

        >>> list1 = [1, 2, 3, 4, 5]
        >>> indices = [1, 3]
        >>> list2 = [6, 7]
        >>> merged_list, new_locs_list1, new_locs_list2 = merge_lists(list1, indices, list2)
        >>> print(merged_list)
        >>> [1, 6, 2, 3, 7, 4, 5]
        >>> print(new_locs_list1)
        >>> {0: 0, 1: 2, 2: 3, 3: 5, 4: 6}
        >>> print(new_locs_list2)
        >>> {0: 1, 1: 4}
        """
        result = []
        i = 0
        j = 0
        index_map_first_list = {}
        index_map_second_list = {}
        while i < len(list1):
            if i in indices:
                result.append(list3[j])
                index_map_second_list[j] = len(result) - 1
                j += 1
            result.append(list1[i])
            index_map_first_list[i] = len(result) - 1
            i += 1
        while j < len(list3):
            result.append(list3[j])
            index_map_second_list[j] = len(result) - 1
            j += 1
        return result, index_map_first_list, index_map_second_list

    def add_shapes(self, shapes, shape_groups, how_many_to_add):
        current_image = self.render_tensor_image_from_shapes(shapes, shape_groups)
        shapes_to_add, shape_groups_to_add = self.get_initial_shapes(how_many_to_add,
                                                                     self.canvas_width,
                                                                     self.canvas_height,
                                                                     current_image)
        # NOTE: for now, we scatter the shapes on top of the existing ones.
        # Consider adding also to background shapes.
        where_to_add = [len(shapes)] * how_many_to_add
        new_shapes, old_shapes_ids_to_new_shape_ids, new_shapes_ids = \
            self.interleave_lists(shapes, where_to_add, shapes_to_add)
        new_shape_groups = []
        for shg in shape_groups:
            new_shape_id = old_shapes_ids_to_new_shape_ids[shg.shape_ids.item()]
            new_shape_groups.append(self.create_shape_group(
                new_shape_id, fill_color=shg.fill_color))
        for shg in shape_groups_to_add:
            new_shape_id = new_shapes_ids[shg.shape_ids.item()]
            new_shape_groups.append(self.create_shape_group(
                new_shape_id, fill_color=shg.fill_color))
        new_shape_groups.sort(key=lambda x: x.shape_ids.item())
        return new_shapes, new_shape_groups

    def run(self):
        # init:
        add_to_file({'1_start': time.time()}, self.timing_file)
        if self.input_svg is None:
            shapes, shape_groups = self.get_initial_shapes(self.shapes_num_scheduler[0],
                                                           self.canvas_width,
                                                           self.canvas_height, )
        else:
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(self.input_svg)
        self.save_intermediate_image(shapes, shape_groups, epoch=-1)
        add_to_file({'1_end': time.time()}, self.timing_file)
        # for loop on reduce steps:
        for epoch, (curr_num_shapes, next_num_shapes, num_iters) in enumerate(zip(
                self.shapes_num_scheduler[:-1], self.shapes_num_scheduler[1:], self.num_iterations[:-1])):
            # optimize:
            add_to_file({f'{epoch+2}_start': time.time()}, self.timing_file)
            shapes, shape_groups = self.optimize_shapes(
                self.image_for_diffvg, shapes, shape_groups,
                num_iters, epoch, early_stopping=self.early_stopping)
            self.save_intermediate_image(shapes, shape_groups, 2 * epoch)
            self.save_svg_image_by_name(shapes, shape_groups,
                                        f"after_optimization_{len(shapes):04d}")
            assert len(shapes) == curr_num_shapes
            # (rank+)reduce or add:
            how_many_to_carve = curr_num_shapes - next_num_shapes - 1
            if how_many_to_carve > 0:  # reduce
                # rank:
                shapes_importance = self.shapes_to_importance(shapes,
                                                              shape_groups)
                assert len(shapes) == curr_num_shapes
                shapes, shape_groups = self.carve_shapes(shapes, shape_groups,
                                                         shapes_importance,
                                                         how_many_to_carve)
                assert len(shapes) == next_num_shapes
            else:  # add shapes
                how_many_to_add = next_num_shapes - curr_num_shapes
                shapes, shape_groups = self.add_shapes(shapes, shape_groups,
                                                       how_many_to_add)
                assert len(shapes) == next_num_shapes
            self.save_intermediate_image(shapes, shape_groups, 2 * epoch + 1)
            add_to_file({f'{epoch+2}_end': time.time()}, self.timing_file)

        # lastly, we need to optimize:
        add_to_file({f'{len(self.shapes_num_scheduler[:-1]) + 2}_start': time.time()}, self.timing_file)
        total_num_epochs = 2 * len(self.shapes_num_scheduler)
        shapes, shape_groups = self.optimize_shapes(
            self.image_for_diffvg, shapes, shape_groups, self.num_iterations[-1],
            2 * total_num_epochs)
        self.save_final_result(shapes, shape_groups)
        add_to_file({f'{len(self.shapes_num_scheduler[:-1]) + 2}_total_end': time.time()}, self.timing_file)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(script_args):
    # set_seed(0)
    image_name = script_args.target.split("/")[-1].split(".")[0]
    scheduler = [int(x) for x in script_args.scheduler]
    iterations = [int(x) for x in script_args.num_iter]
    num_paths = scheduler[0]
    recons_loss_type = script_args.recons_loss_type
    geometric_loss_type = script_args.geometric_loss_type
    if script_args.experiment_name != '':
        experiment_name = script_args.experiment_name
    else:
        experiment_name = f'reduce_or_add_and_optimize_' \
                          f'{image_name}_{num_paths}_' \
                          f'rec_{recons_loss_type}_' \
                          f'geom_{geometric_loss_type}'
        if script_args.advanced_logging:
            experiment_name += "_advanced_logging"

    root_out_dir = osp.join(script_args.results_dir, experiment_name)

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

    diffvg_runner = ReduceOrAddAndOptimize(
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
        init_type=script_args.init_type,
        init_shape=script_args.init_shape,
        timing_json_path=timing_file,
        final_loss_path=final_loss_path,
        early_stopping=script_args.early_stopping,
        sample_beta=script_args.sample_beta,
        input_svg=script_args.input_svg,
    )
    diffvg_runner.run()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
