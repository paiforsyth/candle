#comment
import math

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .context import *
from .estimator import Function
from .nested import *
from .proxy import *

class WeightMaskGroup(ProxyDecorator):
    def __init__(self, layer, child, init_value=1, stochastic=False):
        super().__init__(layer, child)
        self.stochastic = stochastic
        self.masks = self.build_masks(init_value)
        self.frozen = False
        self._flattened_masks = self.masks.reify(flat=True)
        self.cache = Memoizer()
        self._reset_buffers()

    def _reset_buffers(self):
        if not self.stochastic:
            return
        self._frozen_samples = self.concrete_fn().clamp(0, 1).detach().data.reify(flat=True)

    def _build_masks(self, init_value, sizes, randomized_eval=False):
        log_alpha_mean=0.5 #added by Peter
        if self.stochastic:
            self.concrete_fn = HardConcreteFunction.build(self.layer, sizes,log_alpha_mean, randomized_eval=randomized_eval)
            return self.concrete_fn.parameters()
        else:
            return Package([nn.Parameter(init_value * torch.ones(sizes))])

    def buffers(self):
        if self.stochastic:
            return [(f"frozen_samples{i}", sample) for i, sample in enumerate(self._frozen_samples)]
        return []

    @property
    def frozen_samples(self):
        def fetch_samples():
            samples = []
            for i in range(1000000):
                try:
                    samples.append(Variable(getattr(self.layer, f"frozen_samples{i}")))
                except AttributeError:
                    break
            return Package.reshape_into(self.concrete_fn().nested_shape, samples)
        return self.cache("_samples", fetch_samples)

    @frozen_samples.setter
    def frozen_samples(self, samples):
        for i, sample in enumerate(samples.data.reify(flat=True)):
            getattr(self.layer, f"frozen_samples{i}").copy_(sample)
        self.cache.delete("_samples")

    def expand_masks(self):
        raise NotImplementedError

    def build_masks(self, init_value):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError

    @property
    def n_groups(self):
        #params per group
        total_params = sum((self.expand_masks() != 0).float().sum().data[0].reify(flat=True))
        group_params = sum(self.masks.numel().reify(flat=True))
        return float(total_params / group_params)

    def l0_loss(self, lambd):
        if not self.stochastic:
            raise ValueError("Mask group must be in stochastic mode!")
        cdf_gt0 = self.concrete_fn.cdf_gt0()
        return lambd * sum((self.n_groups * cdf_gt0).sum().reify(flat=True))

    def l1_loss_slimming(self, lambd):
        if self.stochastic:
            raise ValueError("Mask group cannot be in stochastic mode")
        return 0 #only batchnorm weight groups contribute to l1 loss in the network slimming scheme

    def parameters(self):
        return self._flattened_masks

    def sample_concrete(self):
        if not self.stochastic:
            raise ValueError("Mask group must be in stochastic mode!")
        return self.frozen_samples if self.frozen else self.concrete_fn().clamp(0, 1)

    @property
    def mask_unpruned(self):
        if self.stochastic:
             int_package =(self.sample_concrete() != 0).long().sum().data[0]
        else:
          int_package = (self.masks  != 0).long().sum().data[0]
        if not isinstance(int_package.reify()[0], int): #compatibility
            int_package =int_package.item() 
        return int_package.reify()

    def freeze(self, refresh=True):
        if not self.stochastic or self.frozen:
            return
        self.frozen = True
        self.frozen_samples = self.concrete_fn().clamp(0, 1).detach()

    def unfreeze(self):
        self.frozen = False

    def print_info(self):
        super().print_info()
        print("{}: {} => {}".format(type(self), self.child.sizes.reify(), self.mask_unpruned))

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        masks = self.expand_masks()
        return input * masks

class HardConcreteFunction(Function):
    """
    From Learning Sparse Neural Networks Through L_0 Regularization 
    Louizos et al. (2018)
    """

    def __init__(self, context, alpha, beta, gamma=-0.1, zeta=1.1, randomized_eval=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.sizes = alpha.size()
        self.context = context
        self.randomized_eval = randomized_eval

    def __call__(self):
        #self.beta.data.clamp_(1E-8, 1E8)
        self.alpha.data.clamp_(1E-8, 1E8)
        if self.context.training or self.randomized_eval:
            u = self.alpha.apply_fn(lambda x: x.clone().uniform_()) #note alpha only used to get the right size here.  The draws are from uniform [0,1]
            s = (u.log() - (1 - u).log() + self.alpha.log()) / (self.beta + 1E-6)
            mask = s.sigmoid() * (self.zeta - self.gamma) + self.gamma
        else:
            mask = self.alpha.log().sigmoid() * (self.zeta - self.gamma) + self.gamma
        return mask

    def cdf_gt0(self): #greater than 0
        return (self.alpha.log() - self.beta * np.log(-self.gamma / self.zeta)).sigmoid()

    def parameters(self):
        return Package([self.alpha]) 
        #return Package([self.alpha, self.beta])

    @classmethod
    def build(cls, context, sizes,log_alpha_mean, **kwargs):
        if not isinstance(sizes, Package):
            sizes = Package([sizes])
        alpha = sizes.apply_fn(lambda x: nn.Parameter(torch.Tensor(x).normal_(log_alpha_mean, 0.01).exp()))
        beta =2/3 #sizes.apply_fn(lambda x: nn.Parameter(torch.Tensor(x).fill_(2 / 3)))
        return cls(context, alpha, beta, **kwargs)

class RNNMask(WeightMaskGroup):
    def __init__(self, layer, child, **kwargs):
        super().__init__(layer, child, **kwargs)
        self.expand_masks()

    def build_masks(self, init_value): # TODO: bidirectional support
        sizes = self.child.sizes.reify()
        self._expand_size = Package([[size[1][0] // size[1][1]] * 4 for size in sizes])
        mask_sizes = [size[1][1] for size in sizes]
        return self._build_masks(init_value, mask_sizes, randomized_eval=True)

    def expand_masks(self):
        def expand_mask(size, mask, expand_size):
            for _ in range(len(size) - 1):
                mask = mask.unsqueeze(0)
            if len(size) == 2:
                return mask.expand(size[1], -1).repeat(1, expand_size).permute(1, 0)
            else:
                return mask.repeat(expand_size)
        if self.stochastic:
            mask = self.sample_concrete().singleton()
        else:
            raise ValueError("Only stochastic masks supported currently!")
        mask_package = Package([[m] * 4 for m in mask.reify()])
        expand_weight = self.child.sizes.apply_fn(expand_mask, mask_package, self._expand_size)
        return expand_weight

class ConvGroupChannel2DMask(WeightMaskGroup): #for zeroing entire groups. e.g. in resnext 
     def __init__(self, layer, child,conv_group_size, **kwargs):
        self.conv_group_size = int(conv_group_size)
        super().__init__(layer, child, **kwargs)

     def build_masks(self, init_value):
        assert self.child.sizes.reify()[0][0] % self.conv_group_size == 0
        return self._build_masks(init_value, self.child.sizes.reify()[0][0]//self.conv_group_size )

     def split(self, root):
        param = root.parameters()[0]
        split_root = param.view(param.size(0), -1).permute(1, 0)
        return Package([split_root])

     def expand_masks(self):
        if self.stochastic:
            mask = self.sample_concrete().singleton()
        else:
            mask = self._flattened_masks[0]
        sizes = self.child.sizes.reify()[0]
        stretched_mask = Variable(mask.data.new(sizes[0]))
        for i in range(mask.size()[0]):
            stretched_mask[ self.conv_group_size*i:self.conv_group_size*(i+1)] = mask[i] 

        expand_weight = stretched_mask.expand(sizes[3], sizes[2], sizes[1], -1).permute(3, 2, 1, 0)
        expand_bias = stretched_mask
        return Package([expand_weight, expand_bias])


class BatchNorm2DMask(WeightMaskGroup):
    def __init__(self, layer , child, **kwargs):
        super().__init__(layer,child,**kwargs)

    def build_masks(self, init_value):
        return self._build_masks(init_value, self.child.sizes.reify()[0][0])
        # One mask for each batch norm param

    def split(self, root):
        #This method is trivial here, but is implemented tos tay consistent with outher weight groups.  It returns a 1 by N matrix whose ith element is the ith scale factor
        param = root.parameters()[0] #get the scale factors
        split_root = param.view(param.size(0), -1).permute(1, 0)
        return Package([split_root])

    def expand_masks(self):
        if self.stochastic:
            mask = self.sample_concrete().singleton()
        else:
            mask = self._flattened_masks[0]
        expand_weight = mask
        expand_bias = Variable(mask.data.new([1])) #Don't mask biases.  Teh bias of a pruned BN channel can simply be factored into the bias of the subsequent conv, if it is not killed by a ReLU
        return Package([expand_weight, expand_bias])

    def l1_loss_slimming(self,lambd):
        return lambd*self.root.parameters()[0].norm(p=1)


class Channel2DMask(WeightMaskGroup):
    def __init__(self, layer, child, **kwargs):
        super().__init__(layer, child, **kwargs)

    def build_masks(self, init_value):
        return self._build_masks(init_value, self.child.sizes.reify()[0][0])

    def split(self, root):
        param = root.parameters()[0]
        split_root = param.view(param.size(0), -1).permute(1, 0)
        return Package([split_root])

    def expand_masks(self):
        if self.stochastic:
            mask = self.sample_concrete().singleton()
        else:
            mask = self._flattened_masks[0]
        sizes = self.child.sizes.reify()[0]
        expand_weight = mask.expand(sizes[3], sizes[2], sizes[1], -1).permute(3, 2, 1, 0)
        expand_bias = mask
        return Package([expand_weight, expand_bias])


class ExternChannel2DMask(WeightMaskGroup):
    '''
    Intended use of this class is to allow a conv2d to correctly count its multiplies when it is followed by a barchnorm that may habe pruned channels.
    I.e. this channel has no masks of its own.  It simply references another set of masks, so that they can be used to calc multiplies
    '''
    def __init__(self, layer, child, following_proxy_bn, **kwargs):
        super().__init__(layer, child, **kwargs)
        self.following_proxy_bn = following_proxy_bn 

    def build_masks(self,init_value):
        return Package([nn.Parameter(torch.Tensor([]))])

    def split(self, root):
        return Package([nn.Parameter(torch.Tensor([]) )])

    def expand_masks(self):
        weights= self.root.parameters()[0] #only for getting the right device
        return Package([Variable( weights.data.new([1])), Variable(weights.data.new([1]))])


class LinearRowMask(WeightMaskGroup):
    def __init__(self, layer, child, **kwargs):
        super().__init__(layer, child, **kwargs)

    def build_masks(self, init_value):
        return self._build_masks(init_value, self.child.sizes.reify()[0][0])

    def split(self, root):
        return Package([root.parameters()[0].permute(1, 0)])

    def expand_masks(self):
        mask = self.sample_concrete().singleton() if self.stochastic else self._flattened_masks[0]
        expand_weight = mask.expand(self.child.sizes.reify()[0][1], -1).permute(1, 0)
        expand_bias = mask
        return Package([expand_weight, expand_bias])



class LinearColMask(WeightMaskGroup):
    def __init__(self, layer, child, **kwargs):
        super().__init__(layer, child, **kwargs)
        self._dummy = nn.Parameter(torch.ones(child.sizes.reify()[1][0]))
        self._flattened_masks.append(self._dummy)

    def build_masks(self, init_value):
        return self._build_masks(init_value, self.child.sizes.reify()[0][1])

    def split(self, root):
        return Package([root.parameters()[0]])

    def expand_masks(self):
        mask = self.concrete_fn().clamp(0, 1).singleton() if self.stochastic else self._flattened_masks[0]
        expand_weight = mask.expand(self.child.sizes.reify()[0][0], -1)
        expand_bias = self._dummy
        return Package([expand_weight, expand_bias])

class WeightMask(ProxyDecorator):
    def __init__(self, layer, child, init_value=1, stochastic=False):
        super().__init__(layer, child)
        def create_mask(size):
            return nn.Parameter(torch.ones(*size) * init_value)
        self.masks = child.sizes.apply_fn(create_mask)
        self._flattened_masks = self.masks.reify(flat=True)
        self.stochastic = stochastic

    def parameters(self):
        return self._flattened_masks

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        if self.stochastic:
            return input * self.masks.clamp(0, 1).bernoulli()
        return input * self.masks

def _group_rank_norm(context, proxies, p=1):
   # import pdb; pdb.set_trace()
    return [proxy.split(proxy.root).norm(p, 0) for proxy in proxies]

def _group_rank_l1(context, proxies):
    return _group_rank_norm(context, proxies, p=1)

def _group_rank_l2(context, proxies):
    return _group_rank_norm(context, proxies, p=2)

def _single_rank_magnitude(context, proxies):
    providers = context.list_providers()
    ranks = [provider.package.abs() for provider in providers]
    return ranks

_single_rank_methods = dict(magnitude=_single_rank_magnitude)
_group_rank_methods = dict(l1_norm=_group_rank_l1, l2_norm=_group_rank_l2)

class PruneContext(Context):
    def __init__(self, stochastic=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        if kwargs.get("active"):
            layer.hook_weight(WeightMask, stochastic=self.stochastic)
        return layer

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMask))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMask))

    def list_model_params(self):
        return self.list_mask_params(inverse=True)


    def count_unpruned_masks(self):
        #this method will be inheritted, and allows
        return sum( float(p.sum().cpu()) for p in self.list_mask_params())



    def count_unpruned(self):
        return sum( float(p.sum().cpu()) for p in self.list_mask_params())

    def clip_all_masks(self):
        for p in self.list_mask_params():
            p.data.clamp_(0, 1)

    def prune(self, percentage, method="magnitude", method_map=_single_rank_methods, mask_type=WeightMask):
        rank_call = method_map[method]
        proxies = self.list_proxies("weight_hook", mask_type)
        weights_list = rank_call(self, proxies)
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
                self._prune_one_mask(weight, mask, percentage)   

    
    def prune_global_smallest(self, percentage, method="magnitude", method_map=_single_rank_methods, mask_type=WeightMask):
        '''
         Idea is to find the globally smallest weights (across layers) and set the corresponding masks to 0 
         only suitable for situations in which the wegiht norms being pruned have comparable magnitudes across channels
         (i.e. network slimming)
        '''
        rank_call = method_map[method]
        proxies = self.list_proxies("weight_hook", mask_type)
        weights_list = rank_call(self, proxies)
        global_weights = None
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
               global_weights = weight if global_weights is  None else torch.cat([global_weights, weight]) 
        global_weights,_=torch.sort(global_weights)
        proportion=percentage/100
        thresh_dex = math.ceil(proportion*global_weights.size(0))
        thresh = float(global_weights[thresh_dex])
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
                _, indices = torch.sort(weight.view(-1)) #unnecesary
                indices = (indices[mask.view(-1)[indices] != 0) & (weight.view(-1)[indicies] <=thresh ])
                if indices.size(0) <= 1:
                    continue
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

               


            
    def _prune_one_mask(self, weight,mask, percentage):
                '''
    given a tensor of magnitudes and a corresponding tensor og masks, prune the masks corresponding to the smallest magnitudes
                '''
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) <= 1:
                    return
                length = math.ceil(ne0_indices.size(0) * percentage / 100)
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

class GroupPruneContext(PruneContext):
    def __init__(self, stochastic=False, frozen=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic
        self.frozen = frozen

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        if isinstance(layer, ProxyConv2d):
            assert layer.weight_provider.sizes.reify()[0][0] % layer.groups == 0
            if layer.groups == 1:
                conv_group_size =-1
            else:
                conv_group_size = layer.weight_provider.sizes.reify()[0][0] / layer.groups
        else:
            conv_group_size = -1
        layer.hook_weight(self.find_mask_type(type(layer), kwargs.get("prune", "out"), conv_group_size = conv_group_size, following_proxy_bn = kwargs.get("following_proxy_bn", None)) , stochastic=self.stochastic)
        return layer

    def l0_loss(self, lambd):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        loss = 0
        for mask in group_masks:
            loss = loss + mask.l0_loss(lambd)
        return loss

    def l1_loss_slimming(self, lambd):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        loss = 0
        for mask in group_masks:
            loss = loss + mask.l1_loss_slimming(lambd)
        return loss



    def freeze(self, refresh=True):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        for mask in group_masks:
            mask.freeze(refresh=refresh)

    def unfreeze(self):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        for mask in group_masks:
            mask.unfreeze()

    def find_mask_type(self, layer_type, prune="out", conv_group_size=-1, following_proxy_bn = None ):
        #import pdb; pdb.set_trace()
        if layer_type == ProxyLinear and prune == "out":
            return LinearRowMask
        elif layer_type == ProxyLinear and prune == "in":
            return LinearColMask
        elif layer_type == ProxyConv2d  and prune == "out" and conv_group_size ==-1:
            return Channel2DMask
        elif layer_type == ProxyConv2d  and prune == "out" and conv_group_size >= 1:
            logging.info("creating ConvGroupMask with conv_group_size="+str(conv_group_size))
            #import pdb; pdb.set_trace()
            construct= functools.partial( ConvGroupChannel2DMask, conv_group_size =  conv_group_size )
            return construct
        elif layer_type == ProxyConv2d and prune == "slim":
             assert following_proxy_bn is not None
             return  functools.partial(ExternChannel2DMask, following_proxy_bn = following_proxy_bn ) 
        elif layer_type == ProxyBatchNorm2d and prune == "slim":
            return BatchNorm2DMask
        elif layer_type == ProxyRNN:
            return RNNMask
        else:
            raise ValueError("Layer type unsupported!")

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMaskGroup))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMaskGroup))

    def count_unpruned(self):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        return sum(sum((m.expand_masks() != 0).float().sum().cpu().data[0].reify(flat=True)) for m in group_masks)

    def prune(self, percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup):
        super().prune(percentage, method, method_map, mask_type)

    def prune_global_smallest(self, percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup):
        super().prune_global_smallest(percentage, method, method_map, mask_type)
