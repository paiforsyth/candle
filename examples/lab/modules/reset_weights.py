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
    elif isinstance(module, nn.BatchNorm2d):
        module.running_mean.zero_()
        module.running_var.fill_(1)
        module.weight.data.uniform_()
        module.bias.data.zero_()
        return
    elif isinstance(module, nn.LeakyReLU) or isinstance(module, nn.ReLU):
        return
    elif isinstance(module, nn.MaxPool2d):
        return
    elif getattr(module,"reset_weights",None) is not None:
        module.reset_weights()
        return
    elif getattr(module,"reset_underlying_weights",None) is not None:
        module.reset_underlying_weights()
        return
    else:
        for sublayer in module:
            reset_weights(sublayer)
        return
        

