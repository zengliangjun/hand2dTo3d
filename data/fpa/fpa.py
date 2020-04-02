import numpy as np
import os
import os.path as osp
import cv2
import random
import data.utils as _utils

class FPADB():

    def __init__(self, _root_path = "/work.data5/First_Person_Action_Benchmark"):
        self.skeleton_root = osp.join(_root_path, 'Hand_pose_annotation_v1')
        self.video_root = osp.join(_root_path, 'Video_files')
        self.items = []
        self._load_db()

    def _load_db(self):
        _subjects = os.listdir(self.skeleton_root)
        _subjects.sort()
        for _subject in _subjects:
            _actions = os.listdir(osp.join(self.skeleton_root, _subject))
            _actions.sort()
            for _action in _actions:
                _seqids = os.listdir(osp.join(self.skeleton_root, _subject, _action))
                _seqids.sort()
                for _seqid in _seqids:
                    _path = osp.join(_subject, _action, _seqid)
                    _skeleton_file = osp.join(self.skeleton_root, _path, "skeleton.txt")
                    if not osp.exists(_skeleton_file):
                        continue
                    _skeleton_vals = np.loadtxt(_skeleton_file)
                    if 0 == len(_skeleton_vals):
                        continue

                    _skeleton_id = _skeleton_vals[:, :1].reshape(_skeleton_vals.shape[0])
                    _skeleton = _skeleton_vals[:, 1:].reshape(_skeleton_vals.shape[0], 21, -1)

                    _invalids = []
                    _invalids_file = osp.join(self.skeleton_root, _path, "invalid3.txt")
                    if osp.exists(_invalids_file):
                        _invalidArray = np.loadtxt(_invalids_file, dtype=float)
                        _invalidArray = np.array(_invalidArray, dtype=int)
                        if 0 == len(_invalidArray.shape):
                            _invalids.append(_invalidArray)
                        else:
                            _invalids = list(_invalidArray)

                    for _idx, _id in enumerate(_skeleton_id):
                        if int(_id) in _invalids:
                            #print(osp.join(self.skeleton_root, _path), 'invalid: ', int(_id))
                            continue

                        self.items.append((_path, int(_id), _skeleton[_idx][self._id_order()]))

                    #return

    def _id_order(self):
        return np.array([ 0, 1, 6, 7, 8,
                          2, 9, 10, 11, 3,
                          12, 13, 14, 4, 15,
                          16, 17, 5, 18, 19,
                          20])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, _idx):
        #_path, _id, _skel = self.items[_idx]
        return self.items[_idx]

cam_extr = np.array(
        [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
         [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
         [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
         [0, 0, 0, 1]])

focal_lengths = [1395.749023, 1395.749268]
center = [935.732544, 540.681030]

focal_lengths_dx = focal_lengths[0] / center[0]
focal_lengths_dy = focal_lengths[1] / center[1]

cam_intr = np.array([[focal_lengths_dx, 0, 1],
                     [0, focal_lengths_dy, 1],
                     [0, 0, 1]])

class FPA():

    _globa_db = None

    def __init__(self, _train = False):
        if FPA._globa_db == None:
            FPA._globa_db = FPADB()

        self.ids = _utils.db_ids(__file__, _train, len(FPA._globa_db))

    def __len__(self):
        return len(self.ids)

    def _camera3d(self, _skel):
        _skel_hom = np.concatenate([_skel, np.ones([_skel.shape[0], 1])], 1)
        return cam_extr.dot(_skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

    def _uv2d(self, _skel3d):
        _skel_hom2d = cam_intr.dot(_skel3d.transpose()).transpose()
        _skel_2d = (_skel_hom2d / _skel_hom2d[:, 2:])[:, :2]
        return _skel_2d

    def _getimg(self, _idx):
        _idx = self.ids[_idx]
        _path, _id, _skel = FPA._globa_db.items[_idx]
        _img_path = osp.join(FPA._globa_db.video_root, "{0}/color/color_{1:0>4d}.jpeg".format(_path, _id))
        _img = cv2.imread(_img_path, cv2.IMREAD_COLOR)
        return _img

    def __getitem__(self, _idx):
        _idx = self.ids[_idx]
        _path, _id, _skel = FPA._globa_db.items[_idx]

        _skel3d = self._camera3d(_skel)
        _uv2d = self._uv2d(_skel3d)
        _camera2d = _uv2d -1

        _skel2d = _uv2d.copy()
        _skel2d[:, 0] = _skel2d[:, 0] * center[0]
        _skel2d[:, 1] = _skel2d[:, 1] * center[1]

        return _camera2d, _skel3d, _skel2d, (focal_lengths_dx, focal_lengths_dy)

    def getitem_side(self, _idx):
        _camera2d, _skel3d, _skel2d, _focal = self[_idx]
        return _camera2d, _skel3d, _skel2d, _focal, 'right' 