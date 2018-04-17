import logging
import math
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
        return "ProxyBatchNorm2d: num_features={}. Weight Proivder:\n {}".format(self.num_features,repr(self.weight_provider))

    def on_forward(self, x):
        #note: F.batch_norm will automatically update running_mean and runnning_var
        weights = self.weight_provider().reify()
        if (weights[0]==0).any():
            pass
        return F.batch_norm(x,self.running_mean, self.running_var,*weights, training=self.training,momentum= self.momentum,eps= self.eps )
    
    def multiplies(self,img_h, img_w, input_channels, unpruned):
        #important note: mixing bn and conv pruning is not supported at the moment
        #in this case we would need to take into account the reductions in input channels caued both by pruning of the batchnorm and pruning of the output of the preceding conv
        from . import prune
        if isinstance(self.weight_provider,prune.BatchNorm2DMask ):
            if unpruned:
                effective_out = self.num_features 
            else:
                effective_out = self.effective_output_channels()
                logging.debug("effective output channels for ProxyBatchNorm2d is {} ".format(effective_out))
        return  0, effective_out, img_h, img_w

    def effective_output_channels(self):
        return  self.weight_provider.mask_unpruned[0]

    def prop_nonzero_masks(self):
        return self.weight_provider.prop_nonzero_masks()

    def reset_underlying_weights(self):
        self.weight_provider.root.parameters[0].fill_(1)
        self.weight_provider.root.parameters[1].fill_(0)
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_masks(self):
        assert not self.weight_provider.stochastic
        from . import prune
        assert isinstance(self.weight_provider, prune.BatchNorm2DMask )
        for param in self.weight_provider.parameters(): 
            param.data.fill_(1)






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
        self.store_input=False
        self.store_output=False #used with pruning methods that required samples of input and output
        self.store_output_grad=False
        self.record_of_input=[]
        self.record_of_output=[]
        self.record_of_output_grad=[] #used by the NVIDIA pruning method
        self.record_of_abs_deriv_sum=0 #used by the NVIDIA pruning method
        self.pruning_normalization_factor=None #used to weight scores assigned to this layer in normalization
        
        self.mults=None #these are for flops regularizated pruning
        self.mults_im1=None
        self.mults_om1=None
        self.flop_reg_term=None



    def __repr__(self):
            s= "{}(conv_fn = {}, stride = {}, padding ={}, dilation = {}, groups = {} ).  Weight provider ={}".format(self.__class__.__name__, self.conv_fn, self.stride, self.padding, self.dilation, self.groups, self.weight_provider )
            return s

    def on_forward(self, x):
        if self.store_input:
            assert not self.training
            self.record_of_input.append(x)

        weights = self.weight_provider().reify()
        out=  self.conv_fn(x, *weights, **self._conv_kwargs)

        if self.store_output:
            if self.store_output_grad:
                out.retain_grad()
            self.record_of_output.append(out)

        return out


    def update_abs_deriv_sum(self):
        #if len(self.record_of_output)!=1:
        assert(len(self.record_of_output)==1)
        self.record_of_abs_deriv_sum+=(self.record_of_output[0].data*self.record_of_output[0].grad.data).mean(3).mean(2).abs().mean(0) # correction:sum over feature map occurs before abs.  sum over batch occurs after
        self.record_of_output=[]


    #added by Peter 
    def effective_output_channels(self, unpruned=False):
        from . import prune
        base_output_channels=self.weight_provider.sizes.reify()[0][0]

        logging.debug("base output channels is "+str(base_output_channels))
        if unpruned:
            logging.debug("unpruned mode enabled.  using base output_channels")
            return base_output_channels
        elif isinstance(self.weight_provider,IdentityProxy):
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
        elif isinstance(self.weight_provider, prune.Filter2DMask):
            if self.weight_provider.following_proxy_conv is None:
                logging.debug("No following proxy. Using base output channels")
                return base_output_channels
            logging.debug("calculating output channels using following proxy")
            return self.weight_provider.following_proxy_conv.unmasked_input_channels()
        else:
            raise Exception("unknown weight provider type")

    def unmasked_input_channels(self):
        from . import prune
        assert isinstance(self.weight_provider, prune.Filter2DMask )
        return self.weight_provider.mask_unpruned[0]


class ProxyConv3d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv3d, **kwargs)

class ProxyConv2d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv2d, **kwargs)
        
    def multiplies(self,img_h, img_w, input_channels, unpruned, reduce_out_by_one=False):
        from . import prune
        w_dim = self.weight_provider.sizes.reify()[0]
        if self.groups == w_dim[0] and self.groups == w_dim[1]:
            logging.debug("depthwise convolution detected.  Input channels= output channels")
            effective_out = input_channels
        else:
            effective_out = self.effective_output_channels(unpruned=unpruned) 
              #img_h*img_w* effective_out * input_channels  *w_dim[2]*w_dim[3]/self.groups
            if isinstance(self.weight_provider, prune.Filter2DMask):
                effective_in = self.weight_provider.mask_unpruned[0]
            else:
                effective_in = input_channels
        if reduce_out_by_one:
            effective_out=effective_out-1
        mults, out_channels, height, width = util.countmult_util.conv2d_mult_compute(img_h, img_w, in_channels=effective_in, out_channels=effective_out, groups=self.groups, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size, dilation=self.dilation)
        
        # for flop regularization
        mults_im1, _,_,_ =  util.countmult_util.conv2d_mult_compute(img_h, img_w, in_channels=effective_in-1, out_channels=effective_out, groups=self.groups, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size, dilation=self.dilation) 
        mults_om1, _,_,_ =  util.countmult_util.conv2d_mult_compute(img_h, img_w, in_channels=effective_in, out_channels=effective_out-1, groups=self.groups, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size, dilation=self.dilation)
        self.mults=mults
        self.mults_im1=mults_im1
        self.mults_om1=mults_om1




        logging.debug("number of mults is {}".format(mults))  #logging.debug("number of mults is {}*{}*{}*{}*{}*{} / {} = {}".format(img_h,img_w,effective_out,input_channels,w_dim[2],w_dim[3],self.groups,mults)  )
        
        return mults, out_channels, height, width



    def reset_underlying_weights(self):
        logging.info("reseting ProxyConv2D weights")
        wparams = self.weight_provider.root.parameters()[0]
        v=wparams.shape[1]*wparams.shape[2]*wparams.shape[3]
        stdv = 1. / math.sqrt(v)
        self.weight_provider.root.parameters()[0].data.uniform_(-stdv, stdv)
        self.weight_provider.root.parameters()[1].data.uniform_(-stdv, stdv)

    def reset_masks(self):
        from . import prune
        if isinstance(self.weight_provider, prune.ExternChannel2DMask):
            return
        assert isinstance(self.weight_provider, prune.Channel2DMask)
        assert not isinstance(self.weight_provider, IdentityProxy)
        assert not self.weight_provider.stochastic
        for param in self.weight_provider.parameters():
            param.data.fill_(1)


class CondensingConv2d(_ProxyConvNd):
    def __init__(self, weight_provider,num_c_groups, **kwargs):
        super().__init__(weight_provider, F.Conv2d, **kwargs)
        self.num_c_groups = num_c_groups
        self.register_buffer("c_stage",torch.Tensor([0]))
    
    def reset_underlying_weights(self):
        wparams = self.weight_provider.root.parameters()[0]
        v=wparams.shape[1]*wparams.shape[2]*wparams.shape[3]
        stdv = 1. / math.sqrt(v)
        self.weight_provider.root.parameters()[0].data.uniform_(-stdv, stdv)
        self.weight_provider.root.parameters()[1].data.uniform_(-stdv, stdv)

    def reset_masks(self):
        from . import prune
        if isinstance(self.weight_provider, prune.ExternChannel2DMask):
            return
        assert not isinstance(self.weight_provider, IdentityProxy)
        assert not self.weight_provider.stochastic
        for param in self.weight_provider.parameters():
            param.data.fill_(1)


    def condense(self):
        from . import prune
        assert isinstance(self.weight_provider, prune.CondenseMask)
        assert float(self.c_stage) < self.num_c_groups-1
        weights = self.weight_provider.root.parameters()[0]
        total_in_filts = weights.shape[1]
        assert total_in_filts % self.num_c_groups == 0
        num_filts_to_kill = total_in_filts // self.num_c_groups
        self.weight_provider.condense(weights= weights, num_c_groups =self.num_c_groups, num_filts_to_kill = num_filts_to_kill    )




        


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

    def __repr__(self):
        s=super().__repr__()
        s+= " outdim={}, indim={}. Weight_provider{}".format(self.weight_provider.root().reify()[0].shape[0], self.weight_provider.root().reify()[0].shape[1],self.weight_provider )
        return s



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
