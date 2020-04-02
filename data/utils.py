import os
import os.path as osp
import random
import numpy as np

def db_ids(_file, _train, _count):
    _cwd = osp.dirname(osp.abspath(_file))

    if _train:
        _id_files = osp.join(_cwd, 'train.txt')
        _start = 0
        _end = _count * 7 // 8
    else:
        _id_files = osp.join(_cwd, 'test.txt')
        _start = _count * 7 // 8
        _end = _count

    if osp.exists(_id_files):
        _ids = np.loadtxt(_id_files, dtype=int)
        if len(_ids) == _end - _start:
            return _ids
        os.remove(_id_files)

    _ids = [_id for _id in range(_count)]
    random.shuffle(_ids)

    if _start == 0:
        _mid = _end
    else:
        _mid = _start

    _train_ids = np.array(_ids[0:_mid], dtype=int)
    _test_ids = np.array(_ids[_mid:_count], dtype=int)

    np.savetxt(osp.join(_cwd, 'train.txt'), _train_ids, fmt='%i')
    np.savetxt(osp.join(_cwd, 'test.txt'), _test_ids, fmt='%i')

    if _train:
        return _train_ids
    else:
        return _test_ids

def _getbbox(_pts):
    _minx = int(min(_pts[:, 0]))
    _maxx = int(max(_pts[:, 0]))
    _miny = int(min(_pts[:, 1]))
    _maxy = int(max(_pts[:, 1]))
    _centx = (_minx + _maxx) / 2
    _centy = (_miny + _maxy) / 2
    _half_width = (_maxx - _minx) / 2  * 1.3
    _half_height = (_maxy - _miny) / 2  * 1.3
    _half = max(_half_width, _half_height)

    _minx = int(_centx - _half)
    _maxx = int(_centx + _half)
    _miny = int(_centy - _half)
    _maxy = int(_centy + _half)
    return (_minx, _miny, _maxx, _maxy)

def _links():
    return [[0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]]
