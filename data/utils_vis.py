from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data.utils as _utils

def vis_skel3d1(_ax3d, _skel3d, _color = None):
    for _link in _utils._links():
        _line = _skel3d[list(_link)]
        _line = _line.transpose()

        if _color is None:
            _ax3d.scatter(_line[0], _line[1], _line[2])
            _ax3d.plot(_line[0], _line[1], _line[2])
        else:
            _ax3d.scatter(_line[0], _line[1], _line[2], color=_color)
            _ax3d.plot(_line[0], _line[1], _line[2], color=_color)

    _ax3d.scatter(_skel3d[9:10, 0], _skel3d[9:10, 1], _skel3d[9:10, 2], color='r')

def vis_skel2d1(_ax2d, _skel2d, _color = None):
    for _link in _utils._links():
        _line = _skel2d[list(_link)]
        _line = _line.transpose()

        if _color is None:
            _ax2d.scatter(_line[0], _line[1])
            _ax2d.plot(_line[0], _line[1])
        else:
            _ax2d.scatter(_line[0], _line[1], color=_color)
            _ax2d.plot(_line[0], _line[1], color=_color)

def vis_predict(_predict3d, _skel3d, _camera2d):
    if len(_predict3d.shape) == 3:
        _predict3d = _predict3d[0]

    if len(_skel3d.shape) == 3:
        _skel3d = _skel3d[0]

    if len(_camera2d.shape) == 3:
        _camera2d = _camera2d[0]

    _fig = plt.figure()

    _ax3d1 = _fig.add_subplot("221", projection='3d')
    vis_skel3d1(_ax3d1, _predict3d, _color='r')
    vis_skel3d1(_ax3d1, _skel3d, _color='b')
    _ax3d1.set_title('predict3d_gt3d')

    _ax2d2 = _fig.add_subplot("222")
    vis_skel2d1(_ax2d2, _camera2d)
    _ax2d2.set_title('gt2d')

    _ax3d3 = _fig.add_subplot("223", projection='3d')
    vis_skel3d1(_ax3d3, _predict3d, _color='r')
    _ax3d3.set_title('predict3d')

    _ax3d4 = _fig.add_subplot("224", projection='3d')
    vis_skel3d1(_ax3d4, _skel3d)
    _ax3d4.set_title('gt3d')

    plt.show()