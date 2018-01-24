"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import torch
from torch.autograd import Variable


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
    if isinstance(layer_param, list):
        for ind, param in enumerate(layer_param):
            layer_param[ind] = str(ind) + '=' + param
    elif isinstance(layer_param, dict):
        param_dict = layer_param
        layer_param = []
        for key, param in param_dict.iteritems():
            layer_param.append(key + '=' + param)

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
    return start, stop, dim_size, axis


def DFS(func):
    if func in visited:
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    father_func = None
    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    child_name = DFS(u[0])
                    bottoms.append(child_name)
                    father_func = u[0]

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
            if layer_type_name == 'Index':
                """ Change layer name only for 'Slice' """
                tops_dict[func] = tops_dict[father_func] + '_' + tops_dict[func]
    elif dst == 'ncnn':
        if layer_type_name in ['Clone', 'SetItem']:
            tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            tops_dict[func] = bottoms[0]
        else:
            tops_dict[func] = name
            if layer_type_name == 'Index':
                """ Chane layer name for 'Slice' """
                tops_dict[func] = tops_dict[father_func] + '_' + tops_dict[func]
            elif hasattr(func, 'next_functions'):
                """ Change bottom layers name for other multi top layers cases """
                for u in func.next_functions:
                    if u[0] is not None:
                        child_type = str(type(u[0]).__name__)
                        if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                            father_func = u[0]
                            if (father_func in multi_tops) and (len(multi_tops[father_func]) > 1):
                                for i in range(len(bottoms)):
                                    if bottoms[i] == tops_dict[father_func]:
                                        bottoms[i] = tops_dict[father_func] + '_' + tops_dict[func]

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

    elif layer_type_name not in ['Index', 'Clone', 'SetItem']:
            """ Debug """
            # if layer_type_name != 'Cmax':
            #     return tops_dict[func]

            layer = convert('', layer_type_name, func)
            link(layer, name, bottoms, [tops_dict[func]])

    """ If func layer has multiple top layers """
    if (func in multi_tops) and (len(multi_tops[func]) > 1):
        if func in slice_point:
            """ Make an extra dummy layer type 'Slice' after func layer, which not exist in pytorch """
            slice_func = torch.autograd.function
            slice_func.axis = axis_dict[func]
            slice_func.slice_point = slice_point[func]
            slice_layer = convert('', 'Slice', slice_func)
            link(slice_layer, tops_dict[func] + '_slicer', [tops_dict[func]], multi_tops[func])
        elif dst == 'ncnn':
            """
            Make 'Split' copy for each top layer respectively
            (only in ncnn, caffe will automatically handle this case)
            """
            copy_func = torch.autograd.function
            split_layer = convert('', 'MultiCopy', copy_func)
            link(split_layer, tops_dict[func] + '_copyer', [tops_dict[func]], multi_tops[func])

    return tops_dict[func]


def FindMultiTops(func):
    """
        Precount nodes with number of tops(indegree)>1,
        which could be Slice or Split(only in ncnn, for making multiple copies)
    """
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
                    child_name = FindMultiTops(u[0])
                    bottoms.append(child_name)

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

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
        elif layer_type_name == 'BatchNorm':
            tops_dict[func] = name + '_' + 'scale'
        else:
            tops_dict[func] = name

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    father_func = u[0]
                    if father_func not in multi_tops:
                        multi_tops[father_func] = []
                    multi_tops[father_func].append(tops_dict[father_func] + '_' + tops_dict[func])

                    if (layer_type == 'IndexBackward') and isinstance(func.index, tuple):
                        if father_func not in slice_point:
                            slice_point[father_func] = []
                        start, stop, dim_size, axis = GetLayerParam_Index(func)

                        """ Persume the visit of Index layers will be ascending """
                        if start > 0:
                            slice_point[father_func].append(start)
                            axis_dict[father_func] = axis

                            """ Last slice """
                            # if stop == dim_size

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

    """ Travel computational graph in backward order """
    """ Need to count number of tops(indegree) of all nodes first"""
    global visited, tops_dict, layer_type_count, dst
    global multi_tops, slice_point, axis_dict

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    multi_tops = dict()
    slice_point = dict()
    axis_dict = dict()
    dst = 'ncnn'

    for out in outputs:
        FindMultiTops(out.grad_fn)

    """ Travel computational graph in backward order """
    global ncnn_net, ncnn_weights, blob_set
    global convert, link
    ncnn_net = []
    ncnn_weights = []
    convert = convert_ncnn
    link = link_ncnn

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    blob_set = set()

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
    """ Need to count number of tops(indegree) of all nodes first """
    global visited, tops_dict, layer_type_count, dst
    global slice_point, multi_tops, axis_dict
    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    slice_point = dict()
    multi_tops = dict()
    axis_dict = dict()
    dst = 'caffe'

    for out in outputs:
        FindMultiTops(out.grad_fn)

    """ Travel computational graph in backward order """
    global caffe_net
    global convert, link
    convert = convert_caffe
    link = link_caffe
    caffe_net = []

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()

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
