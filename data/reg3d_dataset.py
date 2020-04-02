import random
import numpy as np
from torch.utils.data.dataset import Dataset
import cv2
import data.utils as _utils

class Reg3d_dataset(Dataset):
    def __init__(self, cfg, is_train = False):

        self.cfg = cfg
        self.dbs = []
        self.sides = []

        if is_train:
            _dbs = cfg.train_dbs
        else:
            _dbs = cfg.test_dbs

        if not isinstance(_dbs, list):
            self.dbs.append(_dbs(_train = is_train))
        else:
            for _db in _dbs:
                self.dbs.append(_db(_train = is_train))

        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

        self.db_ids = []
        for _dbIdx, _db in enumerate(self.dbs):
            for _id in range(len(_db)):
                self.db_ids.append((_dbIdx, _id))

        random.shuffle(self.db_ids)

    def __len__(self):
        return len(self.db_ids)

    def __getitem__(self, _index):
        _dbIdx, _id = self.db_ids[_index]

        _db = self.dbs[_dbIdx]
        _camera2d, _skel3d, _skel2d, (focal_lengths_dx, focal_lengths_dy), _side = _db.getitem_side(_id)

        _camera2d[:, 0] = _camera2d[:, 0] / focal_lengths_dx
        _camera2d[:, 1] = _camera2d[:, 1] / focal_lengths_dy

        if _side == 'left':
            _side = 0
        else:
            _side = 1

        '''
        _fig = plt.figure(2)
        _utils_vis.vis_skel3d(_fig, _skel3d)
        _utils_vis.vis_skel2d(_fig, _camera2d)
        '''

        if self.do_augment and random.random() <= 0.5:
            _camera2d[:, 0] = - _camera2d[:, 0]
            _skel3d[:, 0] = - _skel3d[:, 0]

            if 0 == _side:
                _side = 1
            else:
                _side = 0

        return np.array(_skel3d, dtype=np.float32), \
            np.array(_camera2d, dtype=np.float32), _side

class RegDataset(Reg3d_dataset):
    def __init__(self, cfg, is_train = False):
        super(RegDataset, self).__init__(cfg, is_train)

    def __getitem__(self, _index):
        _skel3d, _camera2d, _side = super(RegDataset, self).__getitem__(_index)
        return _skel3d, _camera2d
