import os
import time
from torch.utils.data import DataLoader
import numpy as np
import onnxruntime as ort

from options import args

if __name__ == "__main__":

    _path = "{}/onnx/{}_{}.onnx".format(args.module_path, args.net_type, args.test_epoch)

    ort_session = ort.InferenceSession(_path)
    input_name = ort_session.get_inputs()[0].name
    print(input_name)
    _set = args.dataset_class(args, False)
    _loader = DataLoader(dataset=_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for itr, _data in enumerate(_loader):
        _skel3d, _camera2d = _data

        _skel3d = _skel3d.detach().numpy()
        _camera2d = _camera2d.detach().numpy()

        _predict3d = ort_session.run(None, {input_name: _camera2d})
        _predict3d = _predict3d[0]
        from data.utils_vis import vis_predict
        vis_predict(_predict3d, _skel3d, _camera2d)