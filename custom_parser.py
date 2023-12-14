import argparse
import os.path as osp

SUPPORTED_RECONSTRUCTION_LOSSES = ['mse', 'l1', 'pyramid-mse', 'pyramid-l1',
                                   'clip', 'l1_and_clip_mix']
SUPPORTED_RANKING_LOSSES = SUPPORTED_RECONSTRUCTION_LOSSES + ['histogram']
SUPPORTED_GEOMETRIC_LOSSES = ['none', 'geometric',
                              # 'xing'
                              ]
SUPPORTED_INIT_TYPES = ['random', 'custom']
SUPPORTED_INIT_SHAPES_TYPES = ['circle', 'random']


def parse_arguments():
    parser = argparse.ArgumentParser("Basic DiffVG runner.")
    parser.add_argument("--target", help="target PNG image path",
                        default=osp.join('target_images', 'tiger.png'))
    parser.add_argument("--input_svg", help="initial svg figure",
                        default=None)
    parser.add_argument("--results_dir", help="root results directory",
                        default='results')
    parser.add_argument("--num_paths", type=int, default=512,
                        help='number of bezier curves in final image')
    parser.add_argument("--num_iter", nargs='*', action='store',
                        default=[100, 100, 100, 100, 100, 500],
                        help='number optimization iterations in every epoch')
    parser.add_argument("--num_epochs", type=int, default=1,
                        help='number of epochs, such that totally we have: '
                             'num_epochs * num_iter optimization steps')
    parser.add_argument("--recons_loss_type", type=str, default='l1',
                        choices=SUPPORTED_RECONSTRUCTION_LOSSES)
    parser.add_argument("--geometric_loss_type", type=str, default='geometric',
                        choices=SUPPORTED_GEOMETRIC_LOSSES)
    parser.add_argument("--geometric_loss_lamda_geometric_punish", type=float, default=10.0,
                        help="The punishment for non-convex behaving shapes.")
    parser.add_argument("--lambda_geometric", type=float, default=0.01, help="Lambda geometric loss")
    parser.add_argument("--sample_beta", type=float, default=5e-3, help="Lambda geometric loss")
    parser.add_argument("--l1_and_clip_alpha", type=float, default=1.0,
                        help="The extent to which we take L1 in the convex "
                             "sum of L1&Clip losses.\n"
                             "1.0 = only L1, 0.0 = only Clip")
    parser.add_argument("--clip_config_file", type=str,
                        default='test/config_init.npy',
                        help="The extent to which we take L1 in the convex "
                             "sum of L1&Clip losses.\n"
                             "1.0 = only L1, 0.0 = only Clip")
    parser.add_argument('--scheduler', nargs='*', action='store',
                        default=[256, 128, 64, 32, 16, 8],
                        help='num of shapes schedule in descending order.')
    parser.add_argument("--ranking_loss_type", type=str, default='l1',
                        choices=SUPPORTED_RANKING_LOSSES)
    parser.add_argument("--ranking_l1_and_clip_alpha", type=float, default=1.0,
                        help="The extent to which we take L1 in the convex "
                             "sum of L1&Clip losses for RANKING.\n"
                             "1.0 = only L1, 0.0 = only Clip")
    parser.add_argument("--ranking_clip_config_file", type=str,
                        default='test/config_init.npy',
                        help="clip loss configuration file for RANKING")
    parser.add_argument('--canvas_width', type=int, default=256)
    parser.add_argument('--canvas_height', type=int, default=256)
    parser.add_argument('--advanced_logging', action='store_true',
                        help='generate advanced logging')
    parser.add_argument('--sample_importance', action='store_true',
                        help='Sample shapes during reduce phase')
    parser.add_argument('--experiment_name', type=str,
                        default='',
                        help='specify experiment name, default is:'
                             '{image_name}_{num_shapes}_{rec_loss}_{geom_loss}')
    parser.add_argument('--text_prompt', type=str, default=None, help='Clip text loss')
    parser.add_argument('--init_type', type=str, default='custom',
                        choices=SUPPORTED_INIT_TYPES,
                        help='Shapes initialization method')
    parser.add_argument('--init_shape', type=str, default='circle',
                        choices=SUPPORTED_INIT_SHAPES_TYPES,
                        help='Shapes initialization shape')
    parser.add_argument('--early_stopping', action='store_true',
                        help='if set, we use early stopping.')
    arguments = parser.parse_args()

    assert len(arguments.scheduler) == len(arguments.num_iter)

    return arguments
