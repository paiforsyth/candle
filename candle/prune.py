#comment
import math
import numpy as np
import copy

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
    def __init__(self, layer, child, init_value=1, stochastic=False, init_log_alpha=None):
        self.init_log_alpha=init_log_alpha
        super().__init__(layer, child)
        self.stochastic = stochastic
        self.masks = self.build_masks(init_value)
        self.frozen = False
        self._flattened_masks = self.masks.reify(flat=True)
        self.cache = Memoizer()
        self._reset_buffers()
        self.local_l0_lambd=None #for giving different layers different weights

    #added by Peter
    def __repr__(self):
        s=super().__repr__()
        s+="stochastic={}, local_l0_lambd={}".format(self.stochastic,self.local_l0_lambd)
        return s
 

    def _reset_buffers(self):
        if not self.stochastic:
            return
        self._frozen_samples = self.concrete_fn().clamp(0, 1).detach().data.reify(flat=True)

    def _build_masks(self, init_value, sizes, randomized_eval=False):
        log_alpha_mean=0.5 if self.init_log_alpha == None else self.init_log_alpha #added by Peter
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
        if lambd== None:
            lambd= self.local_l0_lambd
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

    def __repr__(self):
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero= float((self._flattened_masks[0] != 0).sum())
        return "BatchNorm2DMask child={} [{} / {} masks nonzero]".format(self.child,mask_nonzero,mask_len)

    def prop_nonzero_masks(self):
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero= float((self._flattened_masks[0] != 0).sum())
        return mask_nonzero / mask_len

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
        expand_bias = mask # masking biases for debugging
        return Package([expand_weight, expand_bias])

    def l1_loss_slimming(self,lambd):
        return lambd*self.root.parameters()[0].norm(p=1)


class Channel2DMask(WeightMaskGroup):
    def __init__(self, layer, child, **kwargs):
        super().__init__(layer, child, **kwargs)

    def __repr__(self):
        s= super().__repr__()
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero = self.mask_unpruned[0]
       # mask_nonzero= float((self._flattened_masks[0] != 0).long().sum())
        s+= " Nonzero masks: {} / {}".format(mask_nonzero, mask_len)
        return s

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

    def l2_loss_stochastic(self,lambd):
        '''
        Added by Peter
        '''
        if not self.stochastic:
            raise ValueError("Mask group must be in stochastic mode!") 
        cdf_gt0 = self.concrete_fn.cdf_gt0()
        sizes = self.child.sizes.reify()[0]
        expanded_probs = cdf_gt0.reify()[0].expand(sizes[3], sizes[2], sizes[1], -1).permute(3, 2, 1, 0)
        squared_weights=self.root.parameters()[0]*self.root.parameters()[0]
        val= lambd*(squared_weights*expanded_probs).sum()
        return val





class Filter2DMask(WeightMaskGroup):
    '''
    For masking input filters (rows).  Optionall, a refernece to a following convolution can be included.  This allows the convolution making use of the current mask to take any input filter pruing performed by the following convolution into account when computing its output channels
    '''
    def __init__(self, layer, child, following_proxy_conv, **kwargs):
        super().__init__(layer, child, **kwargs)
        self.following_proxy_conv= following_proxy_conv

    def __repr__(self):
        s= super().__repr__()
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero= float((self._flattened_masks[0] != 0).long().sum())
        s+= " Nonzero masks: {} / {}".format(mask_nonzero, mask_len)
        if self.following_proxy_conv is None:
            s+= " Following proxy conv is None"
        else:
            s+= "Following proxy conv is not None"
        return s

    def build_masks(self, init_value):
        return self._build_masks(init_value, self.child.sizes.reify()[0][1]) #number of masks equals number of input channels


    def split(self, root):
        param = root.parameters()[0]
        split_root = param.transpose(1,0).contiguous().view(param.size(1),-1).transpose(1,0).contiguous()
        return Package([split_root])

    def expand_masks(self):
        if self.stochastic:
            mask = self.sample_concrete().singleton()
        else:
            mask =self._flattened_masks[0]
        sizes = self.child.sizes.reify()[0]
        expand_weight=mask.expand(sizes[0],sizes[2],sizes[3],-1 ).permute(0,3,1,2) 
        expand_bias = Variable(expand_weight.data.new([1]))
        return Package([expand_weight, expand_bias])

    
#class Column2DMask(WeightMaskGroup):
#    def __init__(self, layer, child, following_proxy_conv, **kwargs):
#        super().__init__(layer, child, **kwargs)
#        pass


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


    def __repr__(self):
        s= super().__repr__()
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero = self.mask_unpruned[0]
       # mask_nonzero= float((self._flattened_masks[0] != 0).long().sum())
        s+= " Nonzero masks: {} / {}".format(mask_nonzero, mask_len)
        return s

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

    def __repr__(self):
        s= super().__repr__()
        mask_len = self._flattened_masks[0].size(0)
        mask_nonzero = self.mask_unpruned[0]
       # mask_nonzero= float((self._flattened_masks[0] != 0).long().sum())
        s+= " Nonzero masks: {} / {}".format(mask_nonzero, mask_len)
        return s



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

class CondenseMask(WeightMask):
    def __init__(self, layer, child, init_value=1, stochastic=False):
        super().__init__(self,layer,child,init_value,stochastic)
        


    def condense(self, weights ,num_c_groups, num_filts_to_kill):
        '''
        Added by Peter by analogy to condensenet.  Based on the official implementation
        '''
        out_channels = weights.shape[0]
        in_filters = weights.shape[1]
        assert out_channels == self.masks.shape[0]
        assert in_filters == self.masks.shape[1]
        assert weights.shape[2] == weights.shape[3] ==1
        assert out_channels % num_condense_groups == 0
        c_group_size = out_channels // num_condense_groups
        rmasks = self.masks.view(self.masks.shape[0], self.masks.shape[1])

        shuffle_weights = Variable(weights.data.clone()).view(out_channels, in_filters)
        shuffle_weights[rmasks == 0] =float("inf")
        shuffle_weights = shuffle_weights.view(c_group_size, num_c_groups,in_filters)
        shuffle_weights = shuffle_weights.transpose(0,1).contiguous()
        shuffle_weights = shuffle_weights.view(out_channels, in_filters)
        for i in range(num_c_groups):
            grp_i_weights= shuffle_weights[i*c_group_size:(i+1)*c_groups_size,: ]
            grp_i_dex_to_drop = grp_i_weights.abs().sum(0).sort()[1][:num_filts_to_kill]
            for filt in grp_i_dex_to_drop.data:
                self.masks[i::num_c_groups, filt,:,:].fill_(0)

    


def _group_rank_abs_taylor(context, proxies):
    
    out_scores=[]
    for proxy in proxies:
        out_scores.append(Package([proxy.layer.record_of_abs_deriv_sum ]))
    return out_scores

def _group_rank_abs_taylor_normalized(context, proxies):
    out_scores=[]
    for proxy in proxies:
        criterion=proxy.layer.record_of_abs_deriv_sum 
        normalized_criterion = criterion/criterion.norm()
        out_scores.append(Package([criterion]))
    return out_scores

def _group_rank_norm(context, proxies, p=1):
    return [proxy.split(proxy.root).norm(p, 0) for proxy in proxies]

#added by Peter
def _group_rank_random(context, proxies):
    def replace_with_random(var): #given a variable matrix, return a random vector on the same device, with length equal to the number of columns of the matrix
        return Variable(var.data.new(var.shape[1]).uniform_()) 
    return [ proxy.split(proxy.root).apply_fn(replace_with_random)   for proxy in proxies]

def _group_rank_l1(context, proxies):
    return _group_rank_norm(context, proxies, p=1)

def _group_rank_l2(context, proxies):
    return _group_rank_norm(context, proxies, p=2)

def _single_rank_magnitude(context, proxies):
    providers = context.list_providers()
    ranks = [provider.package.abs() for provider in providers]
    return ranks

_single_rank_methods = dict(magnitude=_single_rank_magnitude)
_group_rank_methods = dict(l1_norm=_group_rank_l1, l2_norm=_group_rank_l2, random=_group_rank_random, taylor=_group_rank_abs_taylor)

class PruneContext(Context):
    def __init__(self, stochastic=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        if kwargs.get("active") and not isinstance(layer, CondensingConv2d ):
            layer.hook_weight(WeightMask, stochastic=self.stochastic)
        elif isinstance(layer, CondensingConv2d):
            assert not self.stochastic
            layer.hook_weight(CondenseMask, stochastic=False)
        return layer

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMask))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMask))

    def list_model_params(self):
        return self.list_mask_params(inverse=True)


    #added by Peter
    def count_unpruned_masks(self):
        return sum( float( (p!=0).long().sum().cpu()) for p in self.list_mask_params())


    def count_unpruned(self):
        return sum( float(((p!=0).long()).sum().cpu()) for p in self.list_mask_params())

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


    def prune_proxy_layer(self, layer, provider_type= ProxyDecorator,  percentage=1, method="magnitude", method_map=_single_rank_methods, mask_type=WeightMask):
        '''
        Given a ProxyLAyer, prunes the masks associated witha  provider of a given type by a given(absolute) percentage using a given method
        Returns true if was pruning was sucessful (i.e. there were  enough masks to prune)
        '''

        rank_call = method_map[method]
        assert isinstance(layer, ProxyLayer)
        proxy_list = [layer.find_provider(provider_type)]#typically a list with one element
        weights_list = rank_call(self, proxy_list)
        success_list= []
        for weights, proxy in zip(weights_list, proxy_list): #typically a loop wih one iteration
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()): 
                success_list.append(self._prune_one_mask_absolute_pct(weight, mask, percentage) )  
        return all(success_list)

    
    def hz_lasso_prune(self, proxy_layer, target_num_channels,target_prop, sample_inputs, sample_outputs, solve_for_weights,display=False):
        '''
        sample_inputs: should be a list of batchsize * in_channels * h* w (h and w are img size not kernel size) sample input images to the proxy_layer
        sample_outputs: should be a list of batchsize* out_channels * h*w (h an dw are image isze not kernel size) sample output images.  Generally the advice is that the input images should take into account any prior pruning at earlier layers, but the `output images should not
        ''' 
        assert isinstance(proxy_layer.weight_provider, Filter2DMask)
        starting_channels=proxy_layer.weight_provider.root().reify()[0].size(1)
        logging.info("starting channels is {}".format(starting_channels))
        if target_num_channels is None:
            target_num_channels = math.ceil(starting_channels*target_prop )
        if starting_channels == target_num_channels:
            logging.info("no channels to prune.  returning")
            return

        if proxy_layer.weight_provider.root().reify()[0].is_cuda:
            was_cuda=True
            device=proxy_layer.weight_provider.root().reify()[0].get_device()
        else:
            was_cuda=False

        output_h=sample_outputs[0].shape[2]
        output_w=sample_outputs[0].shape[3]
        input_channels = sample_inputs[0].shape[1]
        def process_input_img_batch(img_batch, weights ,**conv_kwargs):
            '''
            given a batch of input images (as a tensor) and some convolutinal weights and paramers, returns a tesnor T with  dimensions batches by output channels by h by w by input_channels+1. where h and w are the height and width of the image.  the (q,a,b,c,d) entry of this tensor is the contribution of  input channeln d to output channel a at location (b,c) in output image q in the batch.  the final "input channel" is the contribution of the bias term
            '''
            nonlocal output_h
            nonlocal output_w
            nonlocal input_channels
            batch_size = img_batch.shape[0]
            in_h = img_batch.shape[2]
            in_w= img_batch.shape[3]
            output_channels = weights[0].shape[0]
            kernel_h =weights[0].shape[2]
            kernel_w=weights[0].shape[3]

            

            out_tensor = img_batch.data.new(batch_size, output_channels, output_h, output_w, input_channels+1).fill_(float("nan"))
            for i in range(input_channels):
                cur_slice = img_batch.data[:,i,:,:].view(batch_size,1,in_h,in_w)
                cur_weights= weights[0].data[:,i,:,:].view(output_channels,1,kernel_h, kernel_w)
                out_tensor[:,:,:,:,i]=  F.conv2d(cur_slice, cur_weights, bias=None, **conv_kwargs  )
            out_tensor[:,:,:,:,-1]= weights[1].view(output_channels,1,1,1).expand(output_channels,batch_size, output_h,output_w).transpose(1,0) #bias
            assert not (out_tensor ==float("nan")).any()
            return out_tensor
        Btensor = torch.cat([process_input_img_batch(img_batch,  proxy_layer.weight_provider.root().reify(),**proxy_layer._conv_kwargs) for img_batch in sample_inputs ], dim=0 )#dimensions are (num_samples)*(output channels) by h by w by input_channels+1
        Ytensor = torch.cat( sample_outputs, dim=0 ) #dimensions  are num_samples*output_channels by h by w

        Yvec=Ytensor.contiguous().view(-1)
        Bmat =Btensor.contiguous().view(-1, input_channels+1) 

        import sklearn
        from sklearn.linear_model import lasso_path

        np_Yvec= Yvec.cpu().numpy()
        np_Bmat = Bmat.cpu().detach().numpy()
        alphas, coefs, _ =lasso_path(np_Bmat,np_Yvec, n_alphas= min(input_channels,500) )
        nonzero_counts = (coefs!=0).sum(0)
        assert nonzero_counts[0] ==0
        cur_dex=0
        while True:
            cur_dex+=1
            if (cur_dex >= len(nonzero_counts)) or ( nonzero_counts[cur_dex] > target_num_channels+1): #include bias in count
                cur_dex-=1
                break
        
        beta_chosen =torch.Tensor( coefs[:,cur_dex])
        if was_cuda:
            beta_chosen =beta_chosen.cuda(device)

        #update masks
        proxy_layer.weight_provider.masks.reify()[0].data[beta_chosen[:-1]==0]=0

        proxy_layer.weight_provider.root().reify()[0].data*=beta_chosen[:-1].view(1,-1,1,1)
        proxy_layer.weight_provider.root().reify()[1].data*=beta_chosen[-1] #bias

        if not solve_for_weights:
            return
        else:
            iterations=1000
            #layer_clone=copy.deepcopy(proxy_layer)
            Atensor =Variable(torch.cat(sample_inputs, dim=0))
            def least_squares_loss(layer,in_img, out_img):
                return (layer(in_img)-out_img).view(-1).norm()
            logging.info("correcting wieghts after lasso")
            logging.info("initial least squares loss: {}".format(least_squares_loss(proxy_layer,Atensor,Ytensor ) ))
            optimizer=torch.optim.Adam(proxy_layer.weight_provider.root().reify(),lr=0.01)
            from tqdm import tqdm
            optimizer.zero_grad()
            bar=tqdm(range(iterations)) if display else range(iterations)
            ls_loss=float("inf")
            for i in bar:
                oldloss=float(ls_loss)
                ls_loss = least_squares_loss(proxy_layer,Atensor,Ytensor)
                ls_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                dif=oldloss-float(ls_loss)
                if display:
                    bar.set_description("loss={}. dif={}".format(float(ls_loss),dif))
            logging.info("final least squares loss: {}".format(least_squares_loss(proxy_layer,Atensor,Ytensor ) ))







        
            
        
        
        #Yvec= torch.cat([samp.transpose(1,0).view(-1) for samp in sample_outputs ],dim=0) #vector of length outchannels*batchsize*h*w


    
    def prune_global_smallest(self, percentage, method="magnitude", method_map=_single_rank_methods, mask_type=WeightMask, normalize=False):
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
               assert ((mask == 0) | (mask == 1)).all()
               if sum(mask.view(-1)) <=1: 
                   continue # dont include layers with only one unmasked
               local_weight = weight[mask != 0] 
               if normalize:
                   local_weight=local_weight/proxy.layer.pruning_normalization_factor
               global_weights = local_weight if global_weights is  None else torch.cat([global_weights, local_weight ]) 
        if global_weights is None: #no layers with more than one nozero mask
            return
        global_weights,_=torch.sort(global_weights)
        proportion=percentage/100
        thresh_dex = min(math.ceil(proportion*global_weights.size(0)),global_weights.size(0)-1 )
        thresh = float(global_weights[thresh_dex])
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
                local_weight=weight
                import pdb; pdb.set_trace()
                if normalize:
                   local_weight = local_weight/local_weight.pruning_normalization_factor

                _, indices = torch.sort(local_weight.view(-1)) #unnecesary
                if sum(mask.view(-1)) <= 1: #changed 
                    continue #if there is only one nozero mask in this weight group, dont prune it

                indices = indices[(mask.view(-1)[indices] != 0) & (local_weight.view(-1)[indices] <=thresh) ] 

                if indices.size(0) == sum(mask.view(-1)): # we are about to prune them all
                    indices =indices[:-1] #leave one 

                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

               


            
    def _prune_one_mask(self, weight,mask, percentage):

                '''
    given a tensor of magnitudes and a corresponding tensor of masks, prune the masks corresponding to the smallest magnitudes
                '''
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) <= 1:
                    return
                length = math.ceil(ne0_indices.size(0) * percentage / 100)
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0


    def _prune_one_mask_absolute_pct(self, weight,mask, percentage):
                '''
    given a tensor of magnitudes and a corresponding tensor of masks, prune the masks corresponding to the smallest magnitudes
    The number of masks this version prunes is a percentage of the original number of masks, not a percentage of the currently active masks as above.
    returns false if there are not enough masks to prune
                '''
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) <= 1:
                    return False
                length = math.ceil(indices.size(0) * percentage / 100)
                if length > ne0_indices.size(0): #always leave one channel
                    length= ne0_indices.size(0)-1
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0
                    return True
                return False



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
        mask_type = self.find_mask_type( type(layer), kwargs.get("prune", "out"),conv_group_size = conv_group_size, following_proxy_bn = kwargs.get("following_proxy_bn", None), following_proxy_conv = kwargs.get("following_proxy_conv",None)) 
        layer.hook_weight(mask_type, stochastic=self.stochastic)
        return layer

    def l0_loss(self, lambd):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        loss = 0
        for mask in group_masks:
            loss = loss + mask.l0_loss(lambd)
        return loss

    def l2_loss_stochastic(self, lambd):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        loss = 0
        for mask in group_masks:
            loss = loss + mask.l2_loss_stochastic(lambd)
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

    def find_mask_type(self, layer_type, prune="out", conv_group_size=-1, following_proxy_bn = None, following_proxy_conv = None ):
        if layer_type == ProxyLinear and prune == "out":
            return LinearRowMask
        elif layer_type == ProxyLinear and prune == "in":
            return LinearColMask
        elif layer_type == ProxyConv2d  and prune == "out" and conv_group_size ==-1:
            return Channel2DMask
        elif layer_type == ProxyConv2d  and prune == "out" and conv_group_size >= 1:
            logging.info("creating ConvGroupMask with conv_group_size="+str(conv_group_size))
            construct= functools.partial( ConvGroupChannel2DMask, conv_group_size =  conv_group_size )
            return construct
        elif layer_type == ProxyConv2d and prune == "slim":
             assert following_proxy_bn is not None
             return  functools.partial(ExternChannel2DMask, following_proxy_bn = following_proxy_bn ) 
        elif layer_type == ProxyConv2d and prune =="in":
            return functools.partial(Filter2DMask, following_proxy_conv = following_proxy_conv)
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


     #added by Peter
    def count_unpruned_masks(self):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        return sum(  sum(m.mask_unpruned)  for m in group_masks )





    def count_unpruned(self):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        return sum(sum((m.expand_masks() != 0).float().sum().cpu().data[0].reify(flat=True)) for m in group_masks)

    def prune(self, percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup):
        super().prune(percentage, method, method_map, mask_type)

    def prune_global_smallest(self, percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup, normalize=False):
        super().prune_global_smallest(percentage, method, method_map, mask_type, normalize=normalize)

    def prune_proxy_layer(self, layer, provider_type,  percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup):
        return super().prune_proxy_layer(layer, provider_type, percentage, method, method_map, mask_type)

