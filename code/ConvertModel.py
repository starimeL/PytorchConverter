"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import torch
from torch.autograd import Variable
from ConvertLayer_ncnn import LayerParameter_ncnn


def link_caffe(layer, name, bottom, top):
    layer.name = name
    for b in bottom:
        layer.bottom.append(b)
    for t in top:
        layer.top.append(t)

    caffe_net.append(layer)


def link_ncnn(layer, name, bottom, top):
    layer_type = layer.type
    layer_param = layer.param
    for ind, param in enumerate(layer_param):
        layer_param[ind] = str(ind) + '=' + param

    pp = []
    pp.append('%-16s' % layer_type)
    pp.append('%-16s %d %d' % (name, len(bottom), len(top)))
    for b in bottom:
        pp.append('%s' % b)
        if b not in blob_set:
            blob_set.add(b)
    for t in top:
        pp.append('%s' % t)
        if t not in blob_set:
            blob_set.add(t)
    layer_param = pp + layer_param

    ncnn_net.append(' '.join(layer_param))

    for w in layer.weights:
        ncnn_weights.append(w)


def GetLayerParam_Index(func):
    for axis, slice_param in enumerate(func.index):
        if isinstance(slice_param, int):
            start = slice_param
            stop = slice_param + 1
        else:
            start = slice_param.start
            stop = slice_param.stop
            step = slice_param.step
        if (start or stop or step) is not None:
            break
    shape = func.input_size
    dim_size = shape[axis]
    return start, stop, dim_size


def DFS(func):
    if func in visited:
        if dst == 'ncnn':
            if (func in split_tops) and (len(split_tops[func]) != 0):
                tops_dict[func] = split_tops[func][0]
                split_tops[func].pop(0)
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    child_name = DFS(u[0])
                    bottoms.append(child_name)

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """ Reaching the root node """
    """  TODO: multi data input """
    if len(bottoms) == 0:
        if 'data' not in layer_type_count:
            layer_type_count['data'] = 1
            """ Gen data layer """
            layer_data = convert('', 'data', inputs)
            link(layer_data, 'data', [], ['data'])

        """ Link it with data input """
        bottoms.append('data')

    """  Skip some pytorch layers  """
    if dst == 'caffe':
        if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem']:
            tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            tops_dict[func] = bottoms[0]
        else:
            tops_dict[func] = name
    elif dst == 'ncnn':
        if layer_type_name in ['Clone', 'SetItem']:
            tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            tops_dict[func] = bottoms[0]
        else:
            tops_dict[func] = name

    """ Split to BatchNorm and Scale """
    if layer_type_name == 'BatchNorm':
        layer_double = convert('', layer_type_name, func)
        scale_name = name + '_' + 'scale'
        if dst == 'caffe':
            link(layer_double[0], name, bottoms, [tops_dict[func]])
            link(layer_double[1], scale_name, [tops_dict[func]], [tops_dict[func]])
        elif dst == 'ncnn':
            link(layer_double[0], name, bottoms, [tops_dict[func]])
            link(layer_double[1], scale_name, [tops_dict[func]], [scale_name])
            tops_dict[func] = scale_name

    elif layer_type_name == 'Index':
        if not isinstance(func.index, tuple):
            return tops_dict[func]

        tops_dict[func] = bottoms[0] + '_' + tops_dict[func]
        if bottoms[0] not in slice_point:
            slice_point[bottoms[0]] = []
            slice_tops[bottoms[0]] = []
        slice_tops[bottoms[0]].append(tops_dict[func])

        start, stop, dim_size = GetLayerParam_Index(func)

        """ Persume the visit of Index layers will be ascending """
        if start > 0:
            slice_point[bottoms[0]].append(start)
            """ Last slice """
            if stop == dim_size:
                func.slice_point = slice_point[bottoms[0]]
                layer = convert('', layer_type_name, func)
                link(layer, bottoms[0] + '_slicer', bottoms, slice_tops[bottoms[0]])

    elif layer_type_name not in ['Clone', 'SetItem']:
            """ Debug """
            # if layer_type_name != 'Cmax':
            #     return tops_dict[func]

            layer = convert('', layer_type_name, func)
            link(layer, name, bottoms, [tops_dict[func]])

    if dst == 'ncnn':
        if (func in split_tops) and (len(split_tops[func]) > 1):
            """ Gen split layer name """
            layer_type_name = 'splitncnn'
            if layer_type_name in layer_type_count:
                layer_type_count[layer_type_name] += 1
            else:
                layer_type_count[layer_type_name] = 1

            name = layer_type_name + '_' + str(layer_type_count[layer_type_name])
            layer_split = LayerParameter_ncnn()
            layer_split.type = 'Split'
            layer_split.param = []
            link(layer_split, name, [tops_dict[func]], split_tops[func])

            tops_dict[func] = split_tops[func][0]
            split_tops[func].pop(0)
        else:
            split_tops[func] = []

    return tops_dict[func]


def FindSplit_ncnn(func):
    if func in visited:
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    child_name = FindSplit_ncnn(u[0])
                    bottoms.append(child_name)

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """  Skip some pytorch layers  """
    if layer_type_name in ['Clone', 'SetItem']:
        tops_dict[func] = bottoms[0]
    elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
        tops_dict[func] = bottoms[0]
    else:
        tops_dict[func] = name

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    if u[0] in split_tops:
                        split_tops[u[0]].append('splitncnn_' + tops_dict[func])
                    else:
                        split_tops[u[0]] = ['splitncnn_' + tops_dict[func]]

    return tops_dict[func]


def ConvertModel_ncnn(pytorch_net, InputShape, softmax=False):
    """ Pytorch to ncnn, only support single tensor input """
    from ConvertLayer_ncnn import convert_ncnn

    """ Need forward once """
    pytorch_net.eval()
    global inputs
    n, c, h, w = InputShape
    inputs = Variable(torch.rand(n, c, h, w), requires_grad=True)
    outputs = pytorch_net(inputs)

    if softmax:
        import torch.nn as nn
        regularize = nn.Softmax()
        outputs = regularize(outputs)

    global ncnn_net, ncnn_weights, visited, tops_dict, layer_type_count, blob_set, dst
    global slice_point, slice_tops
    global split_tops

    """ Travel computational graph in backward order """
    """ Need to count number of tops(indegree) of all nodes first"""
    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    split_tops = dict()
    for out in outputs:
        FindSplit_ncnn(out.grad_fn)

    ncnn_net = []
    ncnn_weights = []
    dst = 'ncnn'

    """ Travel computational graph in backward order """
    global convert, link
    convert = convert_ncnn
    link = link_ncnn
    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    blob_set = set()
    slice_point = dict()
    slice_tops = dict()
    for out in outputs:
        DFS(out.grad_fn)

    text_net = '\n'.join(ncnn_net)
    """ Add layer number and blob number """
    text_net = ('%d %d\n' % (len(ncnn_net), len(blob_set))) + text_net
    """ Add ncnn magic number """
    text_net = '7767517\n' + text_net

    return text_net, ncnn_weights


def ConvertModel_caffe(pytorch_net, InputShape, softmax=False):
    """ Pytorch to Caffe, only support single tensor input """
    import os
    import caffe_pb2 as pb2
    from ConvertLayer_caffe import convert_caffe

    """ Need forward once """
    pytorch_net.eval()
    global inputs
    n, c, h, w = InputShape
    inputs = Variable(torch.rand(n, c, h, w), requires_grad=True)
    outputs = pytorch_net(inputs)

    if softmax:
        import torch.nn as nn
        regularize = nn.Softmax()
        outputs = regularize(outputs)

    """ Travel computational graph in backward order """
    global caffe_net, visited, tops_dict, layer_type_count, dst
    global slice_point, slice_tops
    global convert, link
    convert = convert_caffe
    link = link_caffe
    caffe_net = []
    dst = 'caffe'

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    slice_point = dict()
    slice_tops = dict()
    for out in outputs:
        DFS(out.grad_fn)


    """ Caffe input """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True

    """ Caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for layer in caffe_net:
        binary_weights.layer.extend([layer])

        layer_proto = pb2.LayerParameter()
        layer_proto.CopyFrom(layer)
        del layer_proto.blobs[:]
        text_net.layer.extend([layer_proto])

    return text_net, binary_weights
