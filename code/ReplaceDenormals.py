import numpy as np
import torch


def ReplaceDenormals(net):
    for name, param in net.named_parameters():
        np_arr = param.data.numpy()
        for x in np.nditer(np_arr, op_flags=['readwrite']):
            if abs(x) < 1e-30:
                x[...] = 1e-30
        param.data = torch.from_numpy(np_arr)
