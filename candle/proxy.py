import logging
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nested import Package
import util.countmult_util
class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

class Proxy(object):
    def __init__(self, layer):
        self.child = None
        self.layer = layer

    def parameters(self):
        return []

    def buffers(self):
        return []

    @property
    def root(self):
        return self

    def print_info(self):
        pass


    @property
    def sizes(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class ProxyDecorator(Proxy):
    def __init__(self, layer, child):
        super().__init__(layer)
        self.child = child
        self.layer = layer

    def __repr__(self): 
        s="{}(child = {})".format(self.__class__.__name__,self.child)
        return s

    @property
    def root(self):
        return self.child.root

    def print_info(self):
        self.child.print_info()

    def call(self, package, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.child is not None:
            package = self.child(*args, **kwargs)
            return self.call(package, **kwargs)
        else:
            return self.call(*args, **kwargs)

class FakeProxy(Proxy):
    def __init__(self, layer, parameters):
        super().__init__(layer)
        self.params = list(parameters)

    def parameters(self):
        return self.params

    @property
    def sizes(self):
        return Package([p.size() for p in self.params])

    def __call__(self):
        raise ValueError("FakeProxy not callable!")

class IdentityProxy(Proxy):
    def __init__(self, layer, parameters):
        super().__init__(layer)
        self.package = Package(list(parameters))
        self._flattened_params = self.package.reify(flat=True)

    def parameters(self):
        return self._flattened_params

    @property
    def sizes(self):
        return self.package.size()

    def __call__(self):
        return self.package

class ProxyLayer(nn.Module):
    def __init__(self, weight_provider, registry=None):
        super().__init__()
        self.weight_provider = weight_provider
        self.output_proxy = None
        self.input_proxy = None
        self.registry = registry

        self._param_idx = 0
        self._register_all_params("weight_provider", weight_provider)

    def _register_all_params(self, proxy_type, proxy):
        self.registry.register_proxy(proxy_type, proxy)
        i = 0
        for i, parameter in enumerate(proxy.parameters()):
            self.register_parameter("proxy.{}".format(self._param_idx + i), parameter)
        self._param_idx += i + 1
        for name, buf in proxy.buffers():
            print(name)
            self.register_buffer(name, buf)

    def _find_provider(self, provider_type, provider):
        if isinstance(provider, provider_type):
            return provider
        if isinstance(provider, Proxy):
            return None
        return self._find_provider(provider_type, provider.child)

    def find_provider(self, provider_type):
        return self._find_provider(provider_type, self.weight_provider)

    def hook_weight(self, weight_proxy, **kwargs):
        self.weight_provider = weight_proxy(self, self.weight_provider, **kwargs)
        self._register_all_params("weight_hook", self.weight_provider)
        return self.weight_provider

    def hook_output(self, output_proxy, **kwargs):
        self.output_proxy = output_proxy(self, self.output_proxy, **kwargs)
        self._register_all_params("output_hook", self.output_proxy)
        return self.output_proxy

    def hook_input(self, input_proxy, **kwargs):
        self.input_proxy = input_proxy(self, self.input_proxy, **kwargs)
        self._register_all_params("input_hook", self.input_proxy)
        return self.input_proxy

    def apply_input_hook(self, *args):
        if self.input_proxy is None:
            return args
        return self.input_proxy(*args)

    def apply_output_hook(self, out):
        if self.output_proxy is None:
            return out
        return self.output_proxy(out)

    def forward(self, *args, **kwargs):
        if self.input_proxy is not None:
            args = self.input_proxy(*args)
        out = self.on_forward(*args, **kwargs)
        if self.output_proxy is not None:
            out = self.output_proxy(out)
        
        return out

    def on_forward(self, *args, **kwargs):
        raise NotImplementedError
#commen commentt
class ProxyBatchNorm2d(ProxyLayer):
    def __init__(self,weight_provider, num_features, eps, momentum, **kwargs):
        super().__init__(weight_provider, **kwargs)
        self.num_features = num_features
        self. eps = eps
        self. momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

    def __repr__(self):
        #import pdb; pdb.set_trace()
        return "ProxyBatchNorm2d: num_features={}. Weight Proivder:\n {}".format(self.num_features,repr(self.weight_provider))

    def on_forward(self, x):
        #note: F.batch_norm will automatically update running_mean and runnning_var
        weights = self.weight_provider().reify()
        if (weights[0]==0).any():
            pass
            #import pdb; pdb.set_trace()
        return F.batch_norm(x,self.running_mean, self.running_var,*weights, training=self.training,momentum= self.momentum,eps= self.eps )
    
    def multiplies(self,img_h, img_w, input_channels):
        from . import prune
        if isinstance(self.weight_provider,prune.BatchNorm2DMask ):
            effective_out = self.effective_output_channels()
            logging.debug("effective output channels for ProxyBatchNorm2d is {} ".format(effective_out))
        return  0, effective_out, img_h, img_w

    def effective_output_channels(self):
        return  self.weight_provider.mask_unpruned[0]




class _ProxyConvNd(ProxyLayer):
    def __init__(self, weight_provider, conv_fn, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        super().__init__(weight_provider, **kwargs)
        sizes = weight_provider.sizes.reify()
        self.bias = len(sizes) == 2
        self.kernel_size = sizes[0][2:]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_fn = conv_fn
        self._conv_kwargs = dict(dilation=dilation, padding=padding, stride=stride,groups=groups)
        if not self.bias:
            self._conv_kwargs["bias"] = None
        def __repr__(self):
            s= "{}(conv_fn = {}, stride = {}, padding ={}, dilation = {}, groups = {} ).  Weight provider =".format(self.__class__.__name__, conv_fn, stride, padding, dilation, groups,self.weight_provider )
            return s

    def on_forward(self, x):
        weights = self.weight_provider().reify()
        return self.conv_fn(x, *weights, **self._conv_kwargs)

    #added by Peter 
    def effective_output_channels(self):
        from . import prune
        base_output_channels=self.weight_provider.sizes.reify()[0][0]
        logging.debug("base output channels is "+str(base_output_channels))
        if isinstance(self.weight_provider,IdentityProxy):
            logging.debug("found no weight mask. using base output_channels. ")
            return base_output_channels 
        elif isinstance(self.weight_provider, prune.Channel2DMask):
            effective_out = self.weight_provider.mask_unpruned[0] 
            logging.debug(" effective output channels is "+str(effective_out))
            return effective_out
        elif isinstance(self.weight_provider, prune.ConvGroupChannel2DMask):
            unpruned_masks = self.weight_provider.mask_unpruned[0] 
            effective_out = unpruned_masks * self.weight_provider.conv_group_size
            logging.debug(" effective output channels is {}*{}={}".format(unpruned_masks,self.weight_provider.conv_group_size,effective_out) )
            return effective_out
        elif isinstance(self.weight_provider, prune.ExternChannel2DMask):
            assert isinstance(self.weight_provider.following_proxy_bn,  ProxyBatchNorm2d)
            return self.weight_provider.following_proxy_bn.effective_output_channels() 
        else:
            import pdb; pdb.set_trace()
            raise Exception("unknown weight provider type")


class ProxyConv3d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv3d, **kwargs)

class ProxyConv2d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv2d, **kwargs)

    def multiplies(self,img_h, img_w, input_channels):
        w_dim = self.weight_provider.sizes.reify()[0]
        effective_out = self.effective_output_channels() 
              #img_h*img_w* effective_out * input_channels  *w_dim[2]*w_dim[3]/self.groups
        mults, out_channels, height, width = util.countmult_util.conv2d_mult_compute(img_h, img_w, in_channels=input_channels, out_channels=effective_out, groups=self.groups, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size, dilation=self.dilation)
        logging.debug("number of mults is {}".format(mults))  #logging.debug("number of mults is {}*{}*{}*{}*{}*{} / {} = {}".format(img_h,img_w,effective_out,input_channels,w_dim[2],w_dim[3],self.groups,mults)  )
        return mults, out_channels, height, width

class ProxyConv1d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv1d, **kwargs)

class ProxyLinear(ProxyLayer):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, **kwargs)

    def on_forward(self, x):
        weights = self.weight_provider().reify()
        return F.linear(x, *weights)

    def effective_output_dim(self):
        from . import prune
        if isinstance(self.weight_provider,IdentityProxy):
            return self.weight_provider.sizes.reify()[0][0]
        elif isinstance(self.weight_provider, prune.LinearRowMask) and self.weight_provider.stochastic == False:
            effective_out = self.weight_provider.mask_unpruned[0] 
            return effective_out 
        else:
            raise Exception("unknown weight provider type")



    def multiplies(self, effective_input_dim): 
        '''
        '''
        effective_out = self.effective_output_dim()
        return effective_out*effective_input_dim, effective_out


class ProxyRNNBase(nn.modules.rnn.RNNBase):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super().__init__(mode, input_size, hidden_size, num_layers, bias, batch_first,
            dropout, bidirectional)
        self.weights = None

    def _inject(self, weights):
        self.weights = weights

    def _uninject(self):
        self.weights = None

    @property
    def all_weights(self):
        if not self.weights:
            return super().all_weights
        return self.weights

class ProxyRNN(ProxyLayer):
    def __init__(self, child, weight_provider, **kwargs):
        super().__init__(weight_provider, **kwargs)
        self.child = child
        self.child.flatten_parameters = self._null_fn # flatten_parameters hack

    def _null_fn(*args, **kwargs):
        return

    def on_forward(self, x, *args, **kwargs):
        self.child._inject(self.weight_provider().reify())
        try:
            val = self.child(x, *args, **kwargs)
        finally:
            self.child._uninject()
        return val
