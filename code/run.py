"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import os
import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torchvision

from ConvertModel import ConvertModel_caffe
from ConvertModel import ConvertModel_ncnn

from ReplaceDenormals import ReplaceDenormals


""" Import your net structure here """

"""  ResNet  """
os.sys.path.append('../ModelFiles/ResNet')
import resnet

"""  MobileNet  """
os.sys.path.append('../ModelFiles/MobileNet')
from MobileNet import MobileNet

"""  UNet  """
os.sys.path.append('../ModelFiles/UNet')
import UNet

"""  FaceBoxes  """
os.sys.path.append('../ModelFiles/FaceBoxes')
from FaceBoxes import FaceBoxes

"""  Anime Gan  """
os.sys.path.append('../ModelFiles/_netG_1')
import models


def GenModelZoo():
    """  Specify the input shape and model initializing param  """
    return {
        0: (torchvision.models.squeezenet1_1, [1, 3, 224, 224], [True], {}),
        1: (resnet.resnet50, [1, 3, 224, 224], [True], {}),
        2: (torchvision.models.densenet121, [1, 3, 224, 224], [False], {}),
        3: (MobileNet, [1, 3, 224, 224], [], {}),

        17: (models._netG_1, [1, 100, 1, 1], [1, 100, 3, 64, 1], {}),
        18: (FaceBoxes, [1, 3, 224, 224], [], {}),
        20: (UNet.UNet, [1, 3, 64, 64], [2], {}),
    }


"""  Set empty path to use default weight initialization  """
# model_path = '../ModelFiles/ResNet/resnet50.pth'
model_path = ''

ModelZoo = GenModelZoo()
ModelDir = '../ModelFiles/'

"""  Set to caffe or ncnn  """
dst = 'ncnn'

for i in range(18, 19):
    if i not in ModelZoo:
        continue

    ModuleFunc, InputShape, args, kwargs = ModelZoo[i]
    """  Init pytorch model  """
    pytorch_net = ModuleFunc(*args, **kwargs)

    if model_path != '':
        try:
            pytorch_net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        except AttributeError:
            pytorch_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        NetName = str(pytorch_net.__class__.__name__)
        if not os.path.exists(ModelDir + NetName):
            os.makedirs(ModelDir + NetName)
        print 'Saving default weight initialization...'
        torch.save(pytorch_net.state_dict(), ModelDir + NetName + '/' + NetName + '.pth')

    """ Replace denormal weight values(<1e-30), otherwise may increase forward time cost """
    ReplaceDenormals(pytorch_net)

    """  Connnnnnnnvert!  """
    print('Converting...')
    if dst == 'caffe':
        text_net, binary_weights = ConvertModel_caffe(pytorch_net, InputShape, softmax=False)
    elif dst == 'ncnn':
        text_net, binary_weights = ConvertModel_ncnn(pytorch_net, InputShape, softmax=False)

    """  Save files  """
    NetName = str(pytorch_net.__class__.__name__)
    if not os.path.exists(ModelDir + NetName):
        os.makedirs(ModelDir + NetName)
    print('Saving to ' + ModelDir + NetName)

    if dst == 'caffe':
        import google.protobuf.text_format
        with open(ModelDir + NetName + '/' + NetName + '.prototxt', 'w') as f:
            f.write(google.protobuf.text_format.MessageToString(text_net))
        with open(ModelDir + NetName + '/' + NetName + '.caffemodel', 'w') as f:
            f.write(binary_weights.SerializeToString())

    elif dst == 'ncnn':
        import numpy as np
        with open(ModelDir + NetName + '/' + NetName + '.param', 'w') as f:
            f.write(text_net)
        with open(ModelDir + NetName + '/' + NetName + '.bin', 'w') as f:
            for weights in binary_weights:
                for blob in weights:
                    blob_32f = blob.flatten().astype(np.float32)
                    blob_32f.tofile(f)

    print('Converting Done.')

    """  Test & Compare(optional)  """
    # from test import TestAndCompare
    # TestAndCompare(i, pytorch_net, InputShape, 'Addmm_1', UseImage=False)
