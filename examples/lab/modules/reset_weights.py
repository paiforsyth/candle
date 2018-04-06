import torch
import torch.nn as nn
def reset_weights(module):
    import torch
    if isinstance(module, nn.Conv2d):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        n = module.in_channels
        for k in module.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
        return
    elif isinstance(layer, nn.BatchNorm2d):
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.weight.data.uniform_()
        layer.bias.data.zero_()
        return
    elif isinstance(layer, nn.MaxPool2d):
        return
    elif getattr(layer,"reset_weights",None) is not None:
        layer.reset_weights()
        return
    elif getattr(layer,"reset_underlying_weights",None) is not None:
        layer.reset_underlying_weights()
        return
    else:
        for sublayer in layer:
            sublayer.reset_weights()
        return
        

