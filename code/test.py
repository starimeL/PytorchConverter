import os
os.sys.path.append('/home/starimeliu/Documents/caffe/caffe-master/python')

import caffe
import numpy as np
import cv2
import torch.nn as nn
from torchvision import transforms


def PrintLabel(prob):
    labels_filename = '../TestData/ImageNetLabels.txt'
    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    order = prob.argsort()
    for i in range(3):
        print(labels[order[-1 - i]], prob[order[-1 - i]])


def TestCaffe(proto_path, model_path, inputs, LayerCheck, ModelInd):
    net = caffe.Net(proto_path, model_path, caffe.TEST)
    net.blobs['data'].data[...] = inputs
    print('input blob:')
    print(net.blobs['data'].data[...])

    net.forward()

    if LayerCheck == 'Softmax_1':
        PrintLabel(net.blobs[LayerCheck].data[0].flatten())
    else:
        print(net.blobs[LayerCheck].data[0][...].flatten())
        if (ModelInd == 17):
            result_img = net.blobs[LayerCheck].data[0] * 255
            result_img = result_img.astype(int)
            result_img = np.transpose(result_img, (1, 2, 0))
            result_img = result_img[..., ::-1]
            cv2.imwrite("AnimeNet_result.png", result_img)
        if (ModelInd == 91):
            result_img = net.blobs[LayerCheck].data[0] * 255
            result_img = result_img.astype(int)
            result_img = np.transpose(result_img, (1, 2, 0))
            result_img = result_img[..., ::-1]
            cv2.imwrite("Upsample_result.png", result_img)


def TestPytorch(net, inputs, LayerCheck):
    from torch.autograd import Variable

    inputs = Variable(inputs, requires_grad=True)

    net.eval()
    outputs = net(inputs)

    if LayerCheck == 'Softmax_1':
        m = nn.Softmax()
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = m(outputs)
        PrintLabel(outputs.data.numpy().flatten())
        result = outputs.data.numpy().flatten()
        print(result.shape)
    else:
        print(outputs.data.numpy().flatten())
        # result = outputs.data.numpy().flatten()


class ColorSpaceTransform(object):
    def __call__(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        img = np.concatenate((img_hsv, img_ycc), 2)

        return img


def TestAndCompare(ModelInd, pytorch_net, InputShape, LayerCheck='Softmax_1', UseImage=False):

    # trans = ColorSpaceTransform()
    # inputs = trans(inputs)

    if UseImage:
        img = '../TestData/2008_000536.jpg'
        # inputs = cv2.imread(img, 0)  # 0 for grayscale
        inputs = cv2.imread(img, 1)  # 1 for color
    else:
        n, c, h, w = InputShape
        if (ModelInd == 17):
            """ mean and standard deviation """
            mu, sigma = 0, 1
            inputs = np.random.normal(mu, sigma, w * h * c).reshape(w, h, c)
        else:
            # inputs = np.linspace(1, w * h * c, w * h * c).reshape(w, h, c)
            inputs = np.random.rand(w, h, c)

    print(inputs.shape)

    scale_factor = 1.0 / 255.0
    if UseImage:
        transform_inputs = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (scale_factor, scale_factor, scale_factor)),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        transform_inputs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (scale_factor, scale_factor, scale_factor)),
        ])

    print('Caffe Output:')
    NetName = str(pytorch_net.__class__.__name__)
    proto_path = '../ModelFiles/' + NetName + '/' + NetName + '.prototxt'
    model_path = '../ModelFiles/' + NetName + '/' + NetName + '.caffemodel'
    inputs_caffe = transform_inputs(inputs).numpy()
    TestCaffe(proto_path, model_path, inputs_caffe, LayerCheck, ModelInd)

    print('')
    print('Pytorch Output:')
    inputs_pytorch = transform_inputs(inputs)
    inputs_pytorch = inputs_pytorch.unsqueeze(0)
    TestPytorch(pytorch_net, inputs_pytorch, LayerCheck)
