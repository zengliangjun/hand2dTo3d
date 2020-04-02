
def pre_main_opts(parser):
    # Selected datasets
    parser.add_argument(
        "--datasets",
        choices=[
            "fpa",
        ],
        nargs="+",
        default=['fpa'],
        help="datasets",
    )

    parser.add_argument(
        "--module_path",
        type=str,
        default=osp.join(osp.dirname(osp.abspath(__file__)), 'models'),
        help="module_path",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default='regnetv2mv2',
        help="training config continue_train",
    )

import os
import os.path as osp

def post_main_opts(config):
    from data.fpa.fpa import FPA
    from data.reg3d_dataset import RegDataset
    config.dataset_class = RegDataset

    _db = []
    for _dataset in config.datasets:
        if _dataset == 'fpa':
            _db.append(FPA)
        else:
            raise ('don\' know db')
    config.test_dbs = _db

    config.test_epoch = 4

import argparse
parser = argparse.ArgumentParser(description='reg')
pre_main_opts(parser)
args = parser.parse_args()
post_main_opts(args)
