import math
import collections
import logging
import random
import copy
from enum import Enum
import functools
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from . import serialmodule
from . import shakedrop_func 
from .countmult import count_approx_multiplies
from .reset_weights import reset_weights
from .reset_masks import reset_masks
from .sample_storage import *
from . import shake_shake
from . import msdnet
from torch.autograd import Variable

from candle.prune import PruneContext, GroupPruneContext

import candle.context
import candle.prune
import candle.quantize
import candle.factorize
   # Used ideas from
        # -pyramidnets by Han et al.
        # -resnext by Xie et al. (aggregated residual transformations) 
        # -snapshot ensembles: train 1, get m for free by Huang et al (2017)
        # -deep pyramidal risudal netowrks with separted stochatic depth by Yamada et al. (2016)
        # -Squeeze and Exictaiton networks by Hu et al 2017
        # -Squeezenet by Iandola et al.
        # -Resnet by He et al.
        # -densenet by Huang et al.
        # -squeeze and excitation networks by Hu et al. 


def add_args(parser):
    parser.add_argument("--squeezenet_in_channels",type=int, default=1)
    parser.add_argument("--squeezenet_base",type=int, default=128)
    parser.add_argument("--squeezenet_incr",type=int, default=128)
    parser.add_argument("--squeezenet_multiplicative_incr", type=int, default=2)
    parser.add_argument("--squeezenet_prop3",type=float, default=0.5)
    parser.add_argument("--squeezenet_freq",type=int, default=2)
    parser.add_argument("--squeezenet_freq_offset",type=int, default=0)
    parser.add_argument("--squeezenet_sr",type=float, default=0.125)
    parser.add_argument("--squeezenet_out_dim",type=int)

    parser.add_argument("--squeezenet_mode",type=str, choices=["msd_fire","zag_fire","shuffle_fire", "bnnfire", "resfire","wide_resfire","dense_fire","dense_fire_v2","next_fire","normal",'mnist_mlp',"mnist_lenet", "mnist_lenet_simp","vgg_fire"], default="normal")

    parser.add_argument("--squeezenet_dropout_rate",type=float,default=0)
    parser.add_argument("--squeezenet_densenet_dropout_rate",type=float,default=0)
    parser.add_argument("--squeezenet_densenet_no_bottleneck",action="store_true")

    parser.add_argument("--fire_skip_mode", type=str, choices=["simple", "none", "zero_pad"], default= "none")
    parser.add_argument("--squeezenet_pool_interval_mode",type=str, choices=["add","multiply"], default="add")
    parser.add_argument("--squeezenet_pool_interval",type=int, default=4)
    parser.add_argument("--squeezenet_num_fires", type=int, default=8)
    parser.add_argument("--squeezenet_conv1_stride", type=int, default=2)
    parser.add_argument("--squeezenet_conv1_size",type=int, default=7)
    parser.add_argument("--squeezenet_num_conv1_filters", type=int, default=96) 
    parser.add_argument("--squeezenet_pooling_count_offset", type=int, default=2) 
    parser.add_argument("--squeezenet_max_pool_size",type=int, default=3)
    parser.add_argument("--squeezenet_disable_pooling",action="store_true")

    parser.add_argument("--squeezenet_dense_k",type=int, default=12)
    parser.add_argument("--squeezenet_dense_fire_depths",type=str, default="default, shallow, std_equal")
    parser.add_argument("--squeezenet_dense_fire_compress_level", type=float, default=0.5 )
    parser.add_argument("--squeezenet_use_excitation",   action="store_true")
    parser.add_argument("--squeezenet_excitation_r", type=int, default=16 )
    parser.add_argument("--squeezenet_next_fire_groups", type=int, default=32)
    parser.add_argument("--squeezenet_local_dropout_rate", type=int, default=0 )
    parser.add_argument("--squeezenet_num_layer_chunks", type=int, default=1) 
    parser.add_argument("--squeezenet_chunk_across_devices", action="store_true")
    parser.add_argument("--squeezenet_layer_chunk_devices",type=int, nargs="+")
    parser.add_argument("--squeezenet_use_non_default_layer_splits",action="store_true")
    parser.add_argument("--squeezenet_layer_splits",type=int, nargs="*")

    
    parser.add_argument("--squeezenet_next_fire_final_bn", action="store_true")    
    parser.add_argument("--squeezenet_next_fire_stochastic_depth", action="store_true")
    parser.add_argument("--squeezenet_next_fire_shakedrop", action="store_true")

    parser.add_argument("--squeezenet_final_fc", action="store_true")# temporarily removed.  seems bugged
    parser.add_argument("--squeezenet_final_size", type=int, default=8)
    
    parser.add_argument("--squeezenet_next_fire_shake_shake", action= "store_true" )
    parser.add_argument("--squeezenet_excitation_shake_shake", action="store_true")


    parser.add_argument("--squeezenet_bnn_pooling", action="store_true")
    parser.add_argument("--squeezenet_bnn_prelu", action="store_true")

    parser.add_argument("--squeezenet_final_mode", choices=["normal","linear"], default="normal", help="use a standard convolution or a special linear layer to end the network") 
    parser.add_argument("--squeezenet_final_side_length", type=int, default=8, help="sidelength of images before final layer.  Used if final_mode is linear") 
    parser.add_argument("--squeezenet_final_act_mode", choices=["enable", "disable",], default="enable", help="should there be an activation with the final conv. ")
    parser.add_argument("--squeezenet_scale_layer",action="store_true")

    parser.add_argument("--squeezenet_shuffle_fire_g1",type=int,default= 8)
    parser.add_argument("--squeezenet_shuffle_fire_g2",type=int,default= 8)
    parser.add_argument("--squeezenet_shuffle_fire_dont_wrap_sepconv", action="store_true")


    parser.add_argument("--squeezenet_bypass_first_last",action="store_true") #do not wrap first and last convolution in squeezenet`
    parser.add_argument("--squeezenet_next_fire_bypass_first_last",action="store_true") #do not wrap first and last convolution in nextfile with proxy contextg`
    parser.add_argument("--squeezenet_next_fire_bypass_first",action="store_true") #do not wrap first  convolution in nextfile with proxy contextg`
    parser.add_argument("--squeezenet_next_fire_bypass_second",action="store_true") #do not wrap second  convolution in nextfile with proxy contextg`
    parser.add_argument("--squeezenet_next_fire_bypass_third",action="store_true") #do not wrap third  convolution in nextfile with proxy contextg`

    #parser.add_argument("--squeezenet_freeze_hard_concrete_for_testing",action="store_true")
    parser.add_argument("--squeezenet_zag_fire_dropout",type=int, default=0.3)
    parser.add_argument("--squeezenet_zag_dont_bypass_last",action="store_true")

    parser.add_argument("--squeezenet_use_forking", action="store_true")
    parser.add_argument("--squeezenet_fork_after_chunks",type=int,nargs="+") #fork after these chunks, to produce multiple output scores.  all output scores returned during training.  Only last during testing
    parser.add_argument("--squeezenet_fork_module",choices=["zag_fire"], default="zag_fire")
    parser.add_argument("--squeezenet_fork_early_exit",action="store_true")
    parser.add_argument("--squeezenet_fork_entropy_threshold",type=float ) #ith element is entropy required to exit after ith chunk Note that the ith chunk will only be considered as a possible eexit if i is in squeezenet_fork_after_chunks

    parser.add_argument("--squeezenet_msd_growth_rate", type=int)
    parser.add_argument("--squeezenet_msd_num_scales",type=int, default=3)
    parser.add_argument("--squeezenet_skip_conv1",action="store_true")

    parser.add_argument("--squeezenet_downsample_via_stride", action="store_true")
    parser.add_argument("--squeezenet_downsample_stride_freq", default=None) #None means use the same freq as the channel increase
    parser.add_argument("--squeezenet_allow_pooling_after_first_fire",action="store_true")

    parser.add_argument("--squeezenet_condense_num_c_groups",type=int, default=8)

    parser.add_argument("--squeezenet_use_mnist_mlp", action="store_true")


FireConfig=collections.namedtuple("FireConfig","in_channels,num_squeeze, num_expand1, num_expand3, skip")
class Fire(serialmodule.SerializableModule):
    @staticmethod 
    def from_configure(configure):
        return Fire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (1,1))
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip = skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)

    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = F.leaky_relu(self.squeezeconv(x)) # batchsize by num_squeeze by height by width
        out = [F.leaky_relu(self.expand1conv(out)), F.leaky_relu(self.expand3conv(out)) ]  #batchsize by num expand1 by height by width and batchsize by num expand1 by height by width
        out = torch.cat(out,dim=1) #batchsize by num_expand1 +num_expand 3
        if self.skip:
            out=out+x
        return out

class ResFire(serialmodule.SerializableModule):
    @staticmethod 
    def from_configure(configure):
        return ResFire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.bn1 = torch.nn.BatchNorm2d(in_channels)
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (1,1))
      self.bn2 = torch.nn.BatchNorm2d(num_squeeze)
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip=skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)
    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = self.bn1(x)
        out = F.leaky_relu(out)
        out = self.squeezeconv(out)
        out = self.bn2(out)
        out = F.leaky_relu(out)
        out= torch.cat( [self.expand1conv(out), self.expand3conv(out)], dim=1  ) 
        if self.skip:
            out=out+x
        return out

class VGGFire(nn.Sequential):
    def __init__(self, in_channels, out_channels, proxy_ctx, proxy_mode):
            super().__init__()
            conv= nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
            conv = conv if proxy_ctx is None else proxy_ctx.wrap(conv) 
            self.add_module("conv", conv)
            self.add_module("relu", nn.ReLU())

    def compute_pruning_normalization_factor(self, mode):


        if mode == PruningNormalizationMode.BY_LAYER:
            for layer in self:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    layer.pruning_normalization_factor = float(layer.record_of_abs_deriv_sum.norm())

        elif mode == PruningNormalizationMode.BY_BLOCK:
            raise Exception("Not compatible with VGG")  
        elif mode == PruningNormalizationMode.NO_NORMALIZATION:
            return
        else:
            raise Exception("Unknown Pruning normalization Mode")

    def submods(self):
        return self.named_children() 








class NextFire(serialmodule.SerializableModule):

    def __init__(self, in_channels, num_squeeze, num_expand,skip,skipmode, groups=32, final_bn=False,stochastic_depth=False, survival_prob=1, shakedrop=False, shake_shake=False, shake_shake_mode= shake_shake.ShakeMode.IMAGE, proxy_ctx=None,proxy_mode = None, bypass_first_last=True, bypass_first=False, bypass_second=False, bypass_third=False,  stride=1):
     super().__init__()
     if bypass_first_last:
         assert not bypass_first and not bypass_second and not bypass_third
     if proxy_mode == "l1reg_context_slimming":
             assert groups ==1
             bn_wrapper = proxy_ctx.wrap
             first_wrapper = proxy_ctx.wrap
             group_wrapper = proxy_ctx.wrap
             last_wrapper  = proxy_ctx.bypass
             layer_dict = collections.OrderedDict()
             

             layer_dict["bn1"] = bn_wrapper(nn.BatchNorm2d(in_channels))
             layer_dict["leaky_reu1" ] = nn.LeakyReLU(inplace=True)

             bn2 = bn_wrapper(nn.BatchNorm2d(num_squeeze))
             squeeze_conv =   first_wrapper(nn.Conv2d(in_channels, num_squeeze, (1,1)), following_proxy_bn=bn2)

             layer_dict["squeeze_conv"] = squeeze_conv 
             layer_dict["bn2"] =  bn2  

             layer_dict["leaky_relu2"] = nn.LeakyReLU(inplace=True)

             bn3= bn_wrapper(nn.BatchNorm2d(num_squeeze))
             group_conv=group_wrapper(nn.Conv2d(num_squeeze, num_squeeze, kernel_size=3, padding=1, groups=groups), following_proxy_bn=bn3)

             layer_dict["group_conv"] = group_conv 
             layer_dict["bn3"] = bn3 

             layer_dict["leaky_relu3"] = nn.LeakyReLU(inplace=True)
             layer_dict["expand_conv"] =  last_wrapper(nn.Conv2d(num_squeeze, num_expand, kernel_size=1,stride=stride))
     elif proxy_mode == "filter_prune_context":
             assert groups == 1
             bn_wrapper = proxy_ctx.bypass
             first_wrapper = proxy_ctx.wrap
             group_wrapper = proxy_ctx.wrap
             last_wrapper = proxy_ctx.wrap

             last_conv = last_wrapper(nn.Conv2d(num_squeeze, num_expand, kernel_size=1, stride=stride))
             group_conv = group_wrapper(nn.Conv2d(num_squeeze, num_squeeze, kernel_size=3,padding=1, groups=groups), following_proxy_conv=last_conv  )
             squeeze_conv = first_wrapper(nn.Conv2d(in_channels, num_squeeze,(1,1)), following_proxy_conv=group_conv )
             layer_dict = collections.OrderedDict()
             layer_dict["bn1"] = bn_wrapper(nn.BatchNorm2d(in_channels))
             layer_dict["leaky_reu1" ] = nn.LeakyReLU(inplace=True)
             layer_dict["squeeze_conv"] = squeeze_conv 
             layer_dict["bn2"] = bn_wrapper(nn.BatchNorm2d(num_squeeze))
             layer_dict["leaky_relu2"] = nn.LeakyReLU(inplace=True)
             layer_dict["group_conv"]=group_conv
             layer_dict["bn3"] = bn_wrapper(nn.BatchNorm2d(num_squeeze))
             layer_dict["leaky_relu3"] = nn.LeakyReLU(inplace=True)
             layer_dict["expand_conv"] = last_conv

     else:     
        if proxy_mode is not None and proxy_mode!="no_context":
            bn_wrapper = proxy_ctx.bypass
            if bypass_first_last:
                first_wrapper = proxy_ctx.bypass
                last_wrapper = proxy_ctx.bypass 
                group_wrapper = proxy_ctx.wrap
            else:
                first_wrapper = proxy_ctx.wrap if not bypass_first else proxy_ctx.bypass
                group_wrapper = proxy_ctx.wrap if not bypass_second else proxy_ctx.bypass
                last_wrapper = proxy_ctx.wrap if not bypass_third else proxy_ctx.bypass
        else:
            bn_wrapper =first_wrapper = last_wrapper =group_wrapper = lambda x:x

        layer_dict = collections.OrderedDict()
        layer_dict["bn1"] = bn_wrapper(nn.BatchNorm2d(in_channels))
        layer_dict["leaky_reu1" ] = nn.LeakyReLU(inplace=True)
        layer_dict["squeeze_conv"] = first_wrapper(nn.Conv2d(in_channels, num_squeeze, (1,1)))
        layer_dict["bn2"] = bn_wrapper(nn.BatchNorm2d(num_squeeze))
        layer_dict["leaky_relu2"] = nn.LeakyReLU(inplace=True)
        layer_dict["group_conv"] = group_wrapper(nn.Conv2d(num_squeeze, num_squeeze, kernel_size=3, padding=1, groups=groups))
        layer_dict["bn3"] = bn_wrapper(nn.BatchNorm2d(num_squeeze))
        layer_dict["leaky_relu3"] = nn.LeakyReLU(inplace=True)
        layer_dict["expand_conv"] =  last_wrapper(nn.Conv2d(num_squeeze, num_expand, kernel_size=1, stride=stride))
        if final_bn:
            logging.info("Making NextFire layer with a final batchnorm")
            layer_dict["final_bn"]=bn_wrapper(nn.BatchNorm2d(num_expand))

             

     self.seq= nn.Sequential(layer_dict)
     self.skip=skip
     self.skipmode=skipmode
     self.in_channels=in_channels
     self.out_channels=num_expand
     self.stochastic_depth=stochastic_depth
     self.survival_prob=survival_prob
     self.shakedrop=shakedrop
     self.shake_shake=shake_shake
     if self.stochastic_depth:
            logging.info("Making NextFire layer with stochastic depth")
            assert self.skip
            assert not self.shakedrop
     if self.shakedrop:
            logging.info("Making NextFire Layer with Shakedrop")
     if self.shake_shake:
            assert proxy_mode is None
            assert not self.shakedrop
            logging.info("Making a NextFire with Shake Shake")
            self.shake_shake_mode=shake_shake_mode
            self.seq2=copy.deepcopy(self.seq)
            init_p(self.seq2)

    def forward(self, x):
        out= self.seq(x)
        if self.stochastic_depth:
            multiplier = Variable(out.data.new(1,out.data.shape[1],1,1).fill_(1))
            if self.training:
                killtop = random.uniform(0,1)>self.survival_prob
                killbottom = random.uniform(0,1)>self.survival_prob
                if killtop:
                    multiplier[:,:(self.out_channels-self.in_channels),:,:]=0
                if killbottom:
                    multiplier[:,(self.out_channels - self.in_channels):,:,:]=0
            else:
                multiplier = multiplier * self.survival_prob
            out=out*multiplier    
        if self.shakedrop:
            out=shakedrop_func.ShakeDrop.apply(out, -1, 1,0, 1, self.survival_prob  )
        if self.shake_shake:
           out2=self.seq2(x)
           alpha, beta= shake_shake.generate_alpha_beta(x,self.shake_shake_mode, self.training)
           out=shake_shake.ShakeFunc.apply(out,out2,alpha,beta) 

        if self.skip:
            if self.skipmode == FireSkipMode.SIMPLE:
                out = out + x
            elif self.skipmode == FireSkipMode.PAD:
                if self.in_channels <self.out_channels:
                   padding = Variable(out.data.new(out.data.shape[0],self.out_channels-self.in_channels,out.data.shape[2],out.data.shape[3]).fill_(0)) 
                   out = out + torch.cat([x,padding],dim=1 )
                elif self.in_channels == self.out_channels:
                   out = out + x
                else:
                    raise Exception("Number of channels cannot shrink")
            else:
                raise Exception("Unknown FireSkipMode")


        return out

    def multiplies(self,img_h, img_w, input_channels, unpruned ):
        assert not self.shake_shake and not self.shakedrop and not self.stochastic_depth
        mults, _,out_h,out_w =  count_approx_multiplies(layer=self.seq, img_h=img_h, img_w=img_w,input_channels=input_channels, unpruned=unpruned)        
        return mults, self.out_channels,out_h ,out_w #residual branch means that we get the full number of out_channels even if we pruned some

    def reset_weights(self):
        logging.info("reseting NextFire weights")
        reset_weights(self.seq)

    def reset_masks(self):
        logging.info("ressetting NextFire masks")
        reset_masks(self.seq)
        

    def prop_nonzero_masks(self):
        prop1 = self.seq.bn1.prop_nonzero_masks()
        prop2 = self.seq.bn2.prop_nonzero_masks()
        prop3 = self.seq.bn3.prop_nonzero_masks()
        return prop1, prop2, prop3 

    def apply_to_subproxies(self, func):
        import candle.proxy
        return_list=[]
        for sublayer in self.seq:
            if isinstance(sublayer, candle.proxy.ProxyLayer):
                return_list.append(func(sublayer))
        return return_list

    def submods(self):
        return self.seq.named_children() 

    def change_store_input(self,val):
        change_store_input(self.seq,val)
        
    def change_store_output(self,val):
        change_store_output(self.seq,val)

    def compute_pruning_normalization_factor(self, mode):


        if mode == PruningNormalizationMode.BY_LAYER:
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    layer.pruning_normalization_factor = float(layer.record_of_abs_deriv_sum.norm())

        elif mode == PruningNormalizationMode.BY_BLOCK:
            total_sq_norm=0    
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    total_sq_norm+=   float(layer.record_of_abs_deriv_sum.norm())**2
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    layer.pruning_normalization_factor =math.sqrt(total_sq_norm)
        elif mode == PruningNormalizationMode.NO_NORMALIZATION:
            return
        else:
            raise Exception("Unknown Pruning normalization Mode")









class MobileResFire(serialmodule.SerializableModule):
    def __init__(self, in_channels, out_channels, proxy_ctx, proxy_ctx_mode, activation):
        layer_dict= collections.OrderedDict
        layer_dict["bn1"] = nn.BatchNorm2d(in_channels)
        layer_dict["actvation1"] = activation()
        layer_dict.conv1= nn.Conv2d(in_channels, out_channels)


class WideResFire(serialmodule.SerializableModule):
    '''
    Like above, but the squeeze layer has 3 by 3 convs
    '''
    @staticmethod 
    def from_configure(configure):
        return WideResFire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip, local_dropout_rate=0)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.bn1 = torch.nn.BatchNorm2d(in_channels)
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (3,3), padding=(1,1))
      self.bn2 = torch.nn.BatchNorm2d(num_squeeze)
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip=skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)
      self.local_dropout_rate=local_dropout_rate

    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = self.bn1(x)
        out = F.leaky_relu(out)
        out = self.squeezeconv(out)
        out = F.dropout(out, p=self.local_dropout_rate)
        out = self.bn2(out)
        out = F.leaky_relu(out)
        out= torch.cat( [self.expand1conv(out), self.expand3conv(out)], dim=1  ) 
        if self.skip:
            out=out+x
        return out


class ZagFire(serialmodule.SerializableModule):
    '''
        Idea is to reproduce Wide ResNet paper
    '''
    def __init__(self, in_channels, out_channels, proxy_ctx, proxy_mode,bypass_last, activation, dropout_rate):
         super().__init__()
         self.in_channels = in_channels
         self.out_channels = out_channels
         self.bypass_last = bypass_last
         if proxy_mode == "l1reg_context_slimming":
             bn_wrapper = proxy_ctx.wrap
             first_wrapper = proxy_ctx.wrap
             last_wrapper  = proxy_ctx.bypass
             #note we need to treat the first conv specially in this case, so it can calc muttiplies based on the following batch norm
         elif proxy_mode is not None and proxy_mode != "no_context":
            bn_wrapper = proxy_ctx.bypass
            if bypass_last:
                first_wrapper = proxy_ctx.wrap
                last_wrapper = proxy_ctx.bypass
            else:
                first_wrapper = last_wrapper = proxy_ctx.wrap
         else:
            bn_wrapper =first_wrapper =last_wrapper = lambda x:x
         layer_dict = collections.OrderedDict()
         layer_dict["bn1"]=bn_wrapper(nn.BatchNorm2d(in_channels))
         layer_dict["activation1"]=activation
         if proxy_mode != "l1reg_context_slimming":
            layer_dict["conv1"]=   first_wrapper(nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=3, padding=1) )
            layer_dict["dropout"]=nn.Dropout(p=dropout_rate,inplace=True)
            layer_dict["bn2"] = bn_wrapper(nn.BatchNorm2d(out_channels)) 
         else: #neccesary to correctly count multiplies with weight slimming 
             bn2 = bn_wrapper(nn.BatchNorm2d(out_channels)) 
             conv1 = first_wrapper(nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=3, padding=1),following_proxy_bn = bn2 )
             layer_dict["conv1"] = conv1
             layer_dict["dropout"] = nn.Dropout(p=dropout_rate,inplace=True)
             layer_dict["bn2"] = bn2


         layer_dict["activation2"] =activation
         layer_dict["conv2"]=last_wrapper(nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, padding=1))
         self.seq=nn.Sequential(layer_dict)

    def forward(self,x):
        out = self.seq(x)
        if self.in_channels <self.out_channels:
            padding = Variable(out.data.new(out.data.shape[0],self.out_channels-self.in_channels,out.data.shape[2],out.data.shape[3]).fill_(0)) 
            out = out + torch.cat([x,padding],dim=1 )
        elif self.in_channels == self.out_channels:
            out = out + x
        return out
    def multiplies(self,img_h, img_w, input_channels, unpruned=False):
        mults, _,out_h,out_w =  count_approx_multiplies(layer = self.seq, img_h=img_h, img_w=img_w, input_channels=input_channels,unpruned=unpruned)        
        
        return mults, self.out_channels,out_h ,out_w #residual branch means that we get the full number of out_channels even if we pruned some


    def compute_pruning_normalization_factor(self, mode):

        if mode == PruningNormalizationMode.BY_LAYER:
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    layer.pruning_normalization_factor = float(layer.record_of_abs_deriv_sum.norm())

        elif mode == PruningNormalizationMode.BY_BLOCK:
            total_sq_norm=0    
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    total_sq_norm+=   float(layer.record_of_abs_deriv_sum.norm())**2
            for layer in self.seq:
                if isinstance(layer, candle.proxy.ProxyConv2d):
                    layer.pruning_normalization_factor =math.sqrt(total_sq_norm)
        elif mode == PruningNormalizationMode.NO_NORMALIZATION:
            return
        else:
            raise Exception("Unknown Pruning normalization Mode")







class FinalZagBlockFire(serialmodule.SerializableModule):
    def __init__(self, in_channels, out_classes, proxy_ctx, activation, dropout_rate):
        suoer().__init__()
        self.out_classes=out_classes
        self.bn = proxy_ctx.bypass(nn.BatchNorm2d(in_channels))
        self.activation=activation()
        self.fc = proxy_ctx.bypass(nn.Linear(in_channels, out_classes))

    def forward(self,x ):
        x=self.bn(x)
        x=self.activation(x)
        x=F.adaptive_avg_pool2d(x,output_size=1)
        x=self.fc(x)
        x=x.view(-1,self.out_classes, 1,1)
        return x


    

    
class ForkFire(serialmodule.SerializableModule):

        def __init__(self, fork_module):
            super().__init__()
            self.fork_module = fork_module

        def forward(self, x):
            return (x, self.fork_module(x)) 
        
        def multiplies(self, img_h, img_w, input_channels):
            return self.fork_module.multiplies(img_h, img_w, input_channels)
                  

class BNNFire(serialmodule.SerializableModule):
    def __init__(self, binarize_ctx, in_channels, out_channels, pool, use_act=True, use_prelu=False):
        super().__init__()
        self.conv = binarize_ctx.wrap(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1) )
        self.bn = binarize_ctx.bypass(nn.BatchNorm2d(out_channels))
        self.use_act = use_act
        if use_act:
            self.act = candle.quantize.BinaryTanh() 
        self.pool=pool
        if pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        self.use_prelu= use_prelu
        if use_prelu:
            self.prelu =  nn.PReLU(num_parameters=out_channels)
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool_layer(x) 
        if self.use_prelu:
            x = self.prelu(x)
        x = self.bn(x)
        if self.use_act:
            s = self.act(x)
        return x


class FireSkipMode(Enum):
    NONE=0
    SIMPLE=1
    PAD=2

class ExcitationFire(serialmodule.SerializableModule):
    def __init__(self,fire_to_wrap, in_channels, out_channels, r, skip, skipmode, fire_to_wrap2=None, shake_shake_enable=False,shake_shake_mode= shake_shake.ShakeMode.IMAGE, proxy_ctx = None, proxy_mode= None):
        super().__init__()
        compressed_dim=max(1,math.floor( in_channels/r  ))
        self.in_channels = in_channels
        self.out_channels = out_channels
        if proxy_ctx is None:
            self.compress=nn.Linear(in_channels, compressed_dim  )
            self.expand=nn.Linear(compressed_dim, out_channels)
        elif proxy_mode == "prune_context":
            self.compress= proxy_ctx.wrap(nn.Linear(in_channels, compressed_dim ))
            self.expand=proxy_ctx.wrap(nn.Linear(compressed_dim, out_channels))
        elif proxy_mode == "l0reg_context":
            self.compress= proxy_ctx.wrap(nn.Linear(in_channels, compressed_dim ))
            self.expand=proxy_ctx.wrap(nn.Linear(compressed_dim, out_channels))
        elif proxy_mode == "group_prune_context":
            self.compress = proxy_ctx.wrap(nn.Linear(in_channels, compressed_dim ))
            self.expand = proxy_ctx.wrap(nn.Linear(compressed_dim, out_channels)) 
        else:
            raise Exception("unknown ctx")
        
        self.wrapped=fire_to_wrap
            


        self.skip=skip
        self.shake_shake_enable=shake_shake_enable
        if skip:
            logging.info("Creating ExcitationFire with skip layer")
        if shake_shake_enable:
            assert fire_to_wrap2 is not None
            logging.info("Creating excitationfire with shake shake")
            self.shake_shake_mode=shake_shake_mode
            self.compress2=nn.Linear(in_channels, compressed_dim  )
            self.wrapped2=fire_to_wrap2
            self.expand2=nn.Linear(compressed_dim, out_channels)
            init_p(self)
        self.skipmode = skipmode

    def forward(self, x):
        z=torch.mean(x,3)
        z=torch.mean(z,2)
        z=self.compress(z)
        z=F.leaky_relu(z)
        z=self.expand(z)
        z=F.sigmoid(z)
        z=torch.unsqueeze(z,2)
        z=torch.unsqueeze(z,3)
        result=z*self.wrapped(x)

        if self.shake_shake_enable:
            z2=torch.mean(x,3)
            z2=torch.mean(z2,2)
            z2=self.compress2(z2)
            z2=F.leaky_relu(z2)
            z2=self.expand2(z2)
            z2=F.sigmoid(z2)
            z2=torch.unsqueeze(z2,2)
            z2=torch.unsqueeze(z2,3)
            result2=z2*self.wrapped2(x)
            alpha, beta= shake_shake.generate_alpha_beta(x,self.shake_shake_mode, self.training)
            result=shake_shake.ShakeFunc.apply(result,result2,alpha,beta) 


            result=result

        if self.skip:
            if self.skipmode == FireSkipMode.SIMPLE:
                result=result+x
            elif self.skipmode == FireSkipMode.PAD:
                if self.in_channels <self.out_channels:
                   padding = Variable(result.data.new(result.data.shape[0],self.out_channels-self.in_channels,result.data.shape[2],result.data.shape[3]).fill_(0)) 
                   result= result + torch.cat([x,padding],dim=1 )
                elif self.in_channels == self.out_channels:
                   result= result + x
                else:
                    raise Exception("Number of channels cannot shrink")
            else:
                raise Exception("Unknown FireSkipMode")
            
        return result

    def multiplies(self, img_h, img_w, input_channels):
        assert not self.shake_shake_enable
        compress_mults, compress_out_dim = self.compress.multiplies(  effective_input_dim = input_channels )
        expand_mults, _ = self.expand.multiplies(effective_input_dim = compress_out_dim)
        wrapped_mults, _,out_h, out_w = count_approx_multiplies(self.wrapped,img_h=img_h, img_w=img_w, input_channels=input_channels)
        return compress_mults + wrapped_mults + expand_mults, self.out_channels,img_h, img_w
    #existance of residual branch implies we get full complement of output channels


class ScaleLayer(serialmodule.SerializableModule):
    def __init__(self,init_val=0.001):
        super().__init__()
        self.scalar = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x):
        return self.scalar*x

#observation: if groups are of size h in both first and final pointwise convolutions, then an output channel depends on h^2 input channels
ShuffleFireSettings = collections.namedtuple("ShuffleFireSettings","in_chan, bottle_chan, out_chan, groups1, groups2,activation, dont_wrap_sepconv")
class ShuffleFire(serialmodule.SerializableModule):

    def __init__(self, settings, ctx = None):
        super().__init__()
        assert(settings.in_chan % settings.groups1 == 0)
        assert(settings.bottle_chan % settings.groups1 == 0 )
        assert(settings.bottle_chan % settings.groups2 == 0 )
        assert(settings.out_chan % settings.groups2 == 0)
        wrap = ctx.wrap if ctx is not None else lambda w: w
        bypass = ctx.bypass if ctx is not None else lambda w: w
        if ctx is None:
            sep_wrap = lambda w : w
        elif settings.dont_wrap_sepconv:
            sep_wrap = ctx.bypass
        else:
            sep_wrap = ctx.wrap

        self.bn1 = bypass( nn.BatchNorm2d(settings.in_chan))
        self.gconv1 = wrap( nn.Conv2d(in_channels=settings.in_chan, out_channels=settings.bottle_chan, kernel_size=1, groups=settings.groups1) )
        self.bn2 = bypass( nn.BatchNorm2d(settings.bottle_chan))
        self.sepconv = sep_wrap(nn.Conv2d(in_channels=settings.bottle_chan, out_channels=settings.bottle_chan, kernel_size=3, padding=1, groups=settings.bottle_chan))
        self.bn3 = bypass(nn.BatchNorm2d(settings.bottle_chan))
        self.gconv2 = wrap(nn.Conv2d(in_channels=settings.bottle_chan, out_channels=settings.out_chan, kernel_size=1, groups=settings.groups2))
        
        self.groups1 = settings.groups1
        self.groups2 = settings.groups2
        self.in_chan = settings.in_chan
        self.out_chan = settings.out_chan
        self.activation = settings.activation

    def forward(self, x):
        shape = x.shape
        out = self.bn1(x)
        out = self.activation(out)
        out = self.gconv1(out)
        if self. groups1 > 1: 
            out = out.view(shape[0],self.groups1,-1,shape[2],shape[3])
            out = out.transpose(2, 1)
            out = out.contiguous().view(shape[0],-1,shape[2],shape[3])

        out = self.bn2(out)
        out = self.activation(out)
        out = self.sepconv(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.gconv2(out)
        if self.in_chan == self.out_chan:
            out=out+x
        else:
            padding = Variable(out.data.new(out.data.shape[0],self.out_chan-self.in_chan,out.data.shape[2],out.data.shape[3]).fill_(0)) 
            out =out + torch.cat([x, padding], dim=1)
        return out


    def multiplies(self, img_h, img_w, input_channels, unpruned ):
        mult, _, img_h, img_w = count_approx_multiplies([self.gconv1, self.sepconv, self.gconv2 ], img_h=img_h, img_w=img_w, input_channels=input_channels, unpruned = unpruned)
        return mult, input_channels, img_h, img_w #residual path restores number of output channels
     




class DenseFire(serialmodule.SerializableModule):
    def __init__(self, k0, num_subunits, k, prop3):
        super().__init__()
        self.subunit_dict=collections.OrderedDict()
        expand3=max(1,math.floor(prop3*k))
        expand1=k-expand3
        for i in range(num_subunits):
            self.subunit_dict["subfire"+str(i)]=ResFire.from_configure(FireConfig(in_channels=k0+k*i, num_squeeze=4*k, num_expand1=expand1, num_expand3=expand3, skip=False   ) )
        for name, unit in self.subunit_dict.items():
            self.add_module(name,unit)

    def forward(self, x):
        xlist=[]
        for i, (name, unit) in  enumerate(self.subunit_dict.items()):
            xlist.append(x)
            x=unit(torch.cat(xlist,dim=1)) #cat along filter dimension
        return x

class DenseFireV2Section(serialmodule.SerializableModule):
    '''
    Based on the more efficient official implementation of Densenet
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    Use a PruneContext to implement condensenet
    '''
    def __init__(self, input_size, k, num_squeeze, dropout_rate, bottleneck, proxy_ctx, proxy_mode ):
        super().__init__()
        if proxy_mode == None or prox_mode == "no_context":
            bn_w = lambda x: x
            conv1_w = lambda x :x
            conv2_w = lambda x :x
        elif proxy_mode == "condense_context":
            bn_wr = proxy_ctx.bypass
            conv1_w = proxy_ctx.wrap
            conv2_w = proxy_ctx.bypass
        else:
            raise Exception("Unsuported Proxy Mode!")

        layerdict = collections.OrderedDict()
        if bottleneck:
            layerdict["batchnorm1"]= bn_w(nn.BatchNorm2d(input_size))
            layerdict["relu1"]=nn.ReLU(inplace=True)
            layerdict["conv1"]=conv1_w(nn.Conv2d(input_size, num_squeeze, kernel_size=1))
            layerdict["batchnorm2"]=bn_w(nn.BatchNorm2d(num_squeeze))
            layerdict["relu2"]=nn.ReLU(inplace=True)
            layerdict["conv2"]=conv2_w(nn.Conv2d(num_squeeze, k, kernel_size=3, padding=1))
        else:
            layerdict["batchnorm"]=bn_w(nn.BatchNorm2d(input_size))
            layerdict["relu"]=nn.ReLU(inplace=True)
            layerdict["conv"]=conv1_w(nn.Conv2d(input_size, k, kernel_size=3, padding=1))
        if dropout_rate>0:
            layerdict["droupout"]=nn.Dropout(p=dropout_rate)
        self.seq=nn.Sequential(layerdict)

    def forward(self, x):
        return torch.cat([x, self.seq(x)], dim = 1  )

        
class DenseFireV2Block(serialmodule.SerializableModule):
    def __init__(self, k0, k, num_subunits, num_squeeze, dropout_rate,bottleneck):
        super().__init__()
        layer_dict=collections.OrderedDict()
        cur_channels=k0
        for i in range(num_subunits):
            layer_dict["section{}".format(i)]=DenseFireV2Section(input_size=cur_channels, k=k, num_squeeze=num_squeeze, dropout_rate = dropout_rate , bottleneck=bottleneck)
            cur_channels+=k
        self.seq=nn.Sequential(layer_dict)

    def forward(self,x):
        return self.seq(x)

class DenseFireV2Transition(serialmodule.SerializableModule):
    '''
    Note: These Transition layers include average pooling
    '''
    def __init__(self, num_in, num_out):
        super().__init__()
        layer_dict = collections.OrderedDict()
        layer_dict["transition_bn"]=nn.BatchNorm2d(num_in)
        layer_dict["transition_relu"]=nn.ReLU(inplace=True)
        layer_dict["transition_conv"]=nn.Conv2d(num_in,num_out, kernel_size=1)
        layer_dict["transition_pool"]=nn.AvgPool2d(kernel_size=2, stride=2)
        self.seq = nn.Sequential(layer_dict)

    def forward(self,x):
        return self.seq(x)


class MnistMLP(serialmodule.SerializableModule):
    def __init__(self, proxy_ctx):
        super().__init__()
        if proxy_ctx == None:
            wrap= lambda x : x
        else:
            wrap =functools.partial(proxy_ctx.wrap, prune = "in" )
        layer_dict = collections.OrderedDict()
        layer_dict["fc1"]= wrap(nn.Linear(784, 300) )
        layer_dict["relu1"]=nn.LeakyReLU()
        layer_dict["fc2"]=wrap(nn.Linear(300,100))
        layer_dict["relu2`"]=nn.LeakyReLU()
        layer_dict["fc3"]=wrap(nn.Linear(100,10))
        self.seq= nn.Sequential(layer_dict)

    def forward(self,x):
        return self.seq(x.view(-1,784)).view(-1,10,1,1)

    def submods(self):
        return self.seq.named_children() 



class MnistLEnet(serialmodule.SerializableModule):
    def __init__(self, proxy_ctx):
        super().__init__()
        if proxy_ctx == None:
            wrap= lambda x : x
        else:
            wrap =proxy_ctx.wrap 
        layer_dict1 = collections.OrderedDict()
        layer_dict2 = collections.OrderedDict()
        layer_dict1["conv1"]=wrap(nn.Conv2d(1,20,kernel_size=5,padding=2 ))#28 by 28 output
        layer_dict1["relu1"]=nn.LeakyReLU()
        layer_dict1["pool1"]=nn.MaxPool2d(kernel_size=2,stride=2) #14 by 14 output
        layer_dict1["conv2"]=wrap(nn.Conv2d(20,50,kernel_size=5,padding=0 )) #10 by 10 output
        layer_dict1["relu2`"]=nn.LeakyReLU()
        layer_dict1["pool2"]=nn.MaxPool2d(kernel_size=2,stride=2) ##5 by 5 output
        self.seq1=nn.Sequential(layer_dict1)

        layer_dict2["fc1"]= wrap(nn.Linear(1250, 800)) #5*5*50=1250
        layer_dict2["relu1"]=nn.LeakyReLU()
        layer_dict2["fc2"]=wrap(nn.Linear(800,500))
        layer_dict2["relu2`"]=nn.LeakyReLU()
        layer_dict2["fc3"]=wrap(nn.Linear(500,10))
        self.seq2= nn.Sequential(layer_dict2)

    def forward(self,x):
        x=self.seq1(x)
        x=x.view(-1,1250)
        x=self.seq2(x).view(-1,10,1,1)
        return x 

    def submods(self):
        import itertools
        return itertools.chain(self.seq1.named_children(), self.seq2.named_children())



class MnistLEnetSimp(serialmodule.SerializableModule):
    def __init__(self, proxy_ctx):
        super().__init__()
        if proxy_ctx == None:
            wrap= lambda x : x
        else:
            wrap =functools.partial(proxy_ctx.wrap )
        layer_dict1 = collections.OrderedDict()
        layer_dict2 = collections.OrderedDict()
        layer_dict1["conv1"]=wrap(nn.Conv2d(1,6,kernel_size=5,padding=2 )) #output 28 by 28
        layer_dict1["relu1"]=nn.LeakyReLU()
        layer_dict1["pool1"]=nn.MaxPool2d(kernel_size=2,stride=2) #output 14 by 14
        layer_dict1["conv2"]=wrap(nn.Conv2d(6,16,kernel_size=5,padding=0 )) #output 10 by 10
        layer_dict1["relu2`"]=nn.LeakyReLU()
        layer_dict1["pool2"]=nn.MaxPool2d(kernel_size=2,stride=2) #output 5 by 5
        self.seq1=nn.Sequential(layer_dict1)


        layer_dict2["fc1"]= wrap(nn.Linear(400, 120) )
        layer_dict2["relu1"]=nn.LeakyReLU()
        layer_dict2["fc2"]=wrap(nn.Linear(120,84))
        layer_dict2["relu2`"]=nn.LeakyReLU()
        layer_dict2["fc3"]=wrap(nn.Linear(84,10))
        self.seq2= nn.Sequential(layer_dict2)

    def forward(self,x):
        x=self.seq1(x)
        x=x.view(-1,400)
        x=self.seq2(x).view(-1,10,1,1)
        return x

    def submods(self):
        import itertools
        return itertools.chain(self.seq1.named_children(), self.seq2.named_children())





SqueezeNetConfig=collections.namedtuple("SqueezeNetConfig","in_channels, base, incr, prop3, freq, sr, out_dim, skipmode,  dropout_rate, num_fires, pool_interval, conv1_stride, conv1_size, pooling_count_offset, num_conv1_filters,  dense_fire_k,  dense_fire_depth_list, dense_fire_compression_level, mode, use_excitation, excitation_r, pool_interval_mode, multiplicative_incr, local_dropout_rate, num_layer_chunks, chunk_across_devices, layer_chunk_devices, next_fire_groups, max_pool_size,densenet_dropout_rate, disable_pooling, next_fire_final_bn, next_fire_stochastic_depth, use_non_default_layer_splits, layer_splits, next_fire_shakedrop, final_fc, final_size, next_fire_shake_shake,excitation_shake_shake, proxy_context_type,bnn_pooling, final_act_mode, scale_layer,bnn_prelu, shuffle_fire_g1, shuffle_fire_g2, shuffle_fire_dont_wrap_sepconv, bypass_first_last,next_fire_bypass_first_last, zag_fire_dropout, create_svd_rank_prop, factorize_use_factors, zag_dont_bypass_last, use_forking, fork_after_chunks, fork_module, fork_early_exit, fork_entropy_threshold, msd_growth_rate, msd_num_scales, skip_conv1, downsample_via_stride, densenet_no_bottleneck, allow_pooling_after_first_fire, freq_offset, condense_num_c_groups, use_mnist_mlp, final_mode, final_side_length, next_fire_bypass_second, next_fire_bypass_first,next_fire_bypass_third ")
class SqueezeNet(serialmodule.SerializableModule):
    '''
        Used ideas from
        -pyramidnets by Han et al.
        -resnext by Xie et al. (aggregated residual transformations) 
        -snapshot ensembles: train 1, get m for free by Huang et al (2017)
        -deep pyramidal risudal netowrks with separted stochatic depth by Yamada et al. (2016)
        -Squeeze and Exictaiton networks by Hu et al 2017
        -Squeezenet by Iandola et al.
        -Resnet by He et al.
        -densenet by Huang et al.
        -squeeze and excitation networks by Hu et al. 
    '''
    @staticmethod
    def default(in_size,out_dim):
        config=SqueezeNetConfig(in_channels=in_size, base=128, incr= 128, prop3=0.5, freq= 2, sr=0.125, out_dim = out_dim)
        return SqueezeNet(config)
    
    @staticmethod
    def from_args(args):
        if args.fire_skip_mode == "simple":
            skipmode = FireSkipMode.SIMPLE
        elif args.fire_skip_mode == "none":
            skipmode = FireSkipMode.NONE
        elif args.fire_skip_mode == "zero_pad":
            skipmode = FireSkipMode.PAD
        if args.squeezenet_dense_fire_depths=="default":
            depthlist=[6, 12, 24, 16]
        elif args.squeezenet_dense_fire_depths=="shallow":
            depthlist=[1, 2, 4, 8]
        elif args.squeezenet_dense_fire_depths=="std_equal":
            depthlist=[11, 11, 11]
        else:
            depthlist=None

        config=SqueezeNetConfig(in_channels=args.squeezenet_in_channels,
                base=args.squeezenet_base, incr= args.squeezenet_incr,
                prop3=args.squeezenet_prop3, freq=args.squeezenet_freq,
                sr= args.squeezenet_sr, out_dim=args.squeezenet_out_dim, skipmode=skipmode,
                dropout_rate=args.squeezenet_dropout_rate,
                num_fires=args.squeezenet_num_fires,
                pool_interval=args.squeezenet_pool_interval,
                conv1_stride=args.squeezenet_conv1_stride,
                conv1_size=args.squeezenet_conv1_size,
                pooling_count_offset=args.squeezenet_pooling_count_offset,
                num_conv1_filters=args.squeezenet_num_conv1_filters,
                mode= args.squeezenet_mode,
                dense_fire_k=args.squeezenet_dense_k,
                dense_fire_depth_list= depthlist,
                dense_fire_compression_level=args.squeezenet_dense_fire_compress_level,
                use_excitation=args.squeezenet_use_excitation,
                excitation_r=args.squeezenet_excitation_r,
                pool_interval_mode=args.squeezenet_pool_interval_mode,
                multiplicative_incr=args.squeezenet_multiplicative_incr,
                local_dropout_rate = args.squeezenet_local_dropout_rate,
                num_layer_chunks = args.squeezenet_num_layer_chunks,
                chunk_across_devices = args.squeezenet_chunk_across_devices,
                layer_chunk_devices = args.squeezenet_layer_chunk_devices,
                next_fire_groups = args.squeezenet_next_fire_groups,
                max_pool_size = args.squeezenet_max_pool_size,
                densenet_dropout_rate = args.squeezenet_densenet_dropout_rate,
                disable_pooling = args.squeezenet_disable_pooling,
                next_fire_final_bn = args.squeezenet_next_fire_final_bn,
                next_fire_stochastic_depth = args.squeezenet_next_fire_stochastic_depth,
                use_non_default_layer_splits = args.squeezenet_use_non_default_layer_splits,
                layer_splits= args.squeezenet_layer_splits,
                next_fire_shakedrop=args.squeezenet_next_fire_shakedrop,
                final_fc=args.squeezenet_final_fc,
                final_size=args.squeezenet_final_size,
                next_fire_shake_shake=args.squeezenet_next_fire_shake_shake,
                excitation_shake_shake=args.squeezenet_excitation_shake_shake,
                proxy_context_type = args.proxy_context_type,
                bnn_pooling = args.squeezenet_bnn_pooling,
                final_act_mode =args.squeezenet_final_act_mode,
                scale_layer = args.squeezenet_scale_layer,
                bnn_prelu = args.squeezenet_bnn_prelu,
                shuffle_fire_g1 = args.squeezenet_shuffle_fire_g1,
                shuffle_fire_g2 = args.squeezenet_shuffle_fire_g2,
                bypass_first_last = args.squeezenet_bypass_first_last,
                next_fire_bypass_first_last=args.squeezenet_next_fire_bypass_first_last,
                #freeze_hard_concrete_for_testing=args.squeezenet_freeze_hard_concrete_for_testing,
                zag_fire_dropout = args.squeezenet_zag_fire_dropout,
                create_svd_rank_prop = args.create_svd_rank_prop,
                factorize_use_factors = args.factorize_use_factors,
                zag_dont_bypass_last =args.squeezenet_zag_dont_bypass_last,
                use_forking = args.squeezenet_use_forking,
                fork_after_chunks =args.squeezenet_fork_after_chunks,
                fork_module =args.squeezenet_fork_module,
                fork_early_exit=args.squeezenet_fork_early_exit,
                fork_entropy_threshold =args.squeezenet_fork_entropy_threshold,
                msd_growth_rate = args.squeezenet_msd_growth_rate,
                msd_num_scales = args.squeezenet_msd_num_scales,
                skip_conv1 = args.squeezenet_skip_conv1,
                downsample_via_stride= args.squeezenet_downsample_via_stride,
                densenet_no_bottleneck = args.squeezenet_densenet_no_bottleneck,
                allow_pooling_after_first_fire = args.squeezenet_allow_pooling_after_first_fire,
                freq_offset = args.squeezenet_freq_offset,
                condense_num_c_groups = args.squeezenet_condense_num_c_groups,
                shuffle_fire_dont_wrap_sepconv = args.squeezenet_shuffle_fire_dont_wrap_sepconv,
                use_mnist_mlp = args.squeezenet_use_mnist_mlp,
                final_mode= args.squeezenet_final_mode,
                final_side_length =args.squeezenet_final_side_length,
                next_fire_bypass_first= args.squeezenet_next_fire_bypass_first,
                next_fire_bypass_second = args.squeezenet_next_fire_bypass_second,
                next_fire_bypass_third=args.squeezenet_next_fire_bypass_third

                )
        return SqueezeNet(config)

    def __init__(self, config):
        super().__init__()
        self.mode =config.mode
        #self.freeze_hard_concrete_for_testing = config.freeze_hard_concrete_for_testing
        self.chunk_across_devices=config.chunk_across_devices
        self.use_forking =config.use_forking
        self.fork_after_chunks=config.fork_after_chunks
        self.fork_early_exit= config.fork_early_exit
        self.num_layer_chunks = config.num_layer_chunks
        if config.use_forking:
            self.calculating_exit_proportions = False
            self.exit_proportions_calculated = False
            self.exit_tallies = [0]*config.num_layer_chunks
            self.total_exits = 0
            self.exit_proportions = [-1] * config.num_layer_chunks
            self.fork_entropy_threshold  = config.fork_entropy_threshold
        if config.downsample_via_stride:
            stride_freq = config.freq
            assert(config.disable_pooling)
        if config.chunk_across_devices:
            assert len(config.layer_chunk_devices ) == config.num_layer_chunks  
            assert config.num_layer_chunks <= torch.cuda.device_count()
            logging.info("found: "+ str(torch.cuda.device_count()) +" cuda devices." )
            self.layer_chunk_devices=config.layer_chunk_devices
        assert(config.skipmode == FireSkipMode.NONE or config.skipmode == FireSkipMode.SIMPLE or config.skipmode == FireSkipMode.PAD)
        if config.mode == "densefire":
            logging.info("Making a dense squeezenet.")
            assert config.num_fires == len(config.dense_fire_depth_list)
        if config.proxy_context_type == "identity_context":
            proxy_ctx = candle.context.Context()
        elif config.proxy_context_type ==  "prune_context":
            proxy_ctx = candle.prune.PruneContext(config=None, active=True ) 
        elif config.proxy_context_type == "group_prune_context":
            proxy_ctx = candle.prune.GroupPruneContext(stochastic=False)
        elif config.proxy_context_type == "l0reg_context":
            proxy_ctx = candle.prune.GroupPruneContext(stochastic=True) 
        elif config.proxy_context_type == "l1reg_context_slimming":
            proxy_ctx = candle.prune.GroupPruneContext(stochastic = False, prune="slim")
        elif config.proxy_context_type == "no_context":
            proxy_ctx = None
        elif config.proxy_context_type == "tanhbinarize_context":
            proxy_ctx = candle.quantize.TanhBinarizeContext() 
        elif config.proxy_context_type == "stdfactorize_context":
            proxy_ctx = candle.factorize.StdFactorizeContext(svd_rank_prop=config.create_svd_rank_prop, use_factors=config.factorize_use_factors)
        elif config.proxy_context_type == "condense_context":
            proxy_ctx = candle.prune.PruneContext(condense=True, num_c_grous =config.condense_num_c_groups )
        elif config.proxy_context_type == "filter_prune_context":
            proxy_ctx = candle.prune.GroupPruneContext(stochastic =False, prune ="in")
        else:
            raise Exception("unknown proxy_context_type")

        self.proxy_ctx=proxy_ctx
        if config.use_mnist_mlp or config.mode == "mnist_mlp":
         self.layer_chunk_1 =nn.Sequential( 
                 collections.OrderedDict([
                 ("mlp",MnistMLP(proxy_ctx))
                 ]))
         self.layer_chunk_list=[]
         self.layer_chunk_list.append(self.layer_chunk_1)
         return
        if config.mode == "mnist_lenet":
         logging.info("building lenet")
         self.layer_chunk_1= nn.Sequential(
         collections.OrderedDict([
         ("lenet",MnistLEnet(proxy_ctx))
         ]))
         self.layer_chunk_list=[]
         self.layer_chunk_list.append(self.layer_chunk_1)
         return
        if config.mode == "mnist_lenet_simp":
         logging.info("building lenet simp")
         self.layer_chunk_1= nn.Sequential(
         collections.OrderedDict([
         ("lenet_simp",MnistLEnetSimp(proxy_ctx))
         ]))
         self.layer_chunk_list=[]
         self.layer_chunk_list.append(self.layer_chunk_1)
         return


        num_fires=config.num_fires #8
        first_layer_num_convs=config.num_conv1_filters
        first_layer_conv_width=config.conv1_size
        first_layer_padding= first_layer_conv_width // 2  

        pool_offset=config.pooling_count_offset
        layer_dict=collections.OrderedDict()
        if not config.skip_conv1:
         self.channel_counts= [first_layer_num_convs]
         if config.mode != "normal":
            if config.proxy_context_type == "no_context":
                layer_dict["conv1"] = nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)
            elif config.bypass_first_last:
                layer_dict["conv1"] =proxy_ctx.bypass( nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride))
            else:
                layer_dict["conv1"] =proxy_ctx.wrap( nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride))

             
         else:
            #old/bugged
            layer_dict=collections.OrderedDict([
            ("conv1", nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)),
            ("conv1relu", nn.LeakyReLU()),
            ("maxpool1", nn.MaxPool2d(kernel_size=3,stride=2))
            ]) 
            self.channel_counts=[ first_layer_num_convs]#initial number of channels entering ith fire layer (labeled i+2 to match paper)
        else:
            self.channel_counts = [config.in_channels]

        num_pool_so_far =0 #for debugging
        num_stride_so_far=0

        for i in range(num_fires):
            if  config.mode != "dense_fire" and config.mode != "dense_fire_v2" and config.mode != "msd_fire":
                if config.pool_interval_mode == "add":
                    e = config.base+math.floor(config.incr*math.floor((i+config.freq_offset)/config.freq))
                elif config.pool_interval_mode == "multiply":
                    e=config.base* (config.multiplicative_incr ** math.floor((i+config.freq_offset)/config.freq)) 
                if config.downsample_via_stride and i!=0 and (i+config.freq_offset) % stride_freq ==0:
                    do_stride_downsample = True
                else:
                    do_stride_downsample = False
                num_squeeze=max(math.floor(config.sr*e),1)
                num_expand3=max(math.floor(config.prop3*e),1)
                num_expand1=e-num_expand3
                if config.skipmode == FireSkipMode.SIMPLE and e == self.channel_counts[i]:
                    skip_here=True
                    logging.debug("Making simple skip layer.")
                elif config.skipmode == FireSkipMode.PAD:
                    skip_here=True
                    if e == self.channel_counts[i]:
                        logging.debug("Padding is enabled, but channel count has not changed.  Simple skipping will occur")
                    else:
                        logging.debug("Making Padding skip layer")
                else:
                    skip_here=False
                if config.mode == "wide_resfire":
                    name="wide_resfire{}".format(i+2)
                    to_add=WideResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here, local_dropout_rate=config.local_dropout_rate ))
                elif config.mode == "resfire":
                    name="resfire{}".format(i+2)
                    to_add=ResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))
                elif config.mode == "next_fire":
                    name = "next_fire{}".format(i+2)
                    survival_prob = 1-0.5*i/num_fires 
                    if do_stride_downsample:
                        stride=2
                        num_stride_so_far+=1
                    else:
                        stride=1
                    to_add = NextFire(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand=e, skip=skip_here, groups=config.next_fire_groups, skipmode=config.skipmode, final_bn=config.next_fire_final_bn, stochastic_depth=config.next_fire_stochastic_depth, survival_prob = survival_prob, shakedrop=config.next_fire_shakedrop, shake_shake= config.next_fire_shake_shake, proxy_ctx=proxy_ctx, proxy_mode = config.proxy_context_type, bypass_first_last = config.next_fire_bypass_first_last, stride= stride, bypass_first= config.next_fire_bypass_first, bypass_second = config.next_fire_bypass_second, bypass_third=config.next_fire_bypass_third )
                    if config.excitation_shake_shake:
                        to_add2= NextFire(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand=e, skip=skip_here, groups=config.next_fire_groups, skipmode=config.skipmode, final_bn=config.next_fire_final_bn, stochastic_depth=config.next_fire_stochastic_depth, survival_prob = survival_prob, shakedrop=config.next_fire_shakedrop, shake_shake= config.next_fire_shake_shake, proxy_ctx=proxy_ctx, proxy_mode = config.proxy_context_type,  bypass_first_last = config.next_fire_bypass_first_last  )
                elif config.mode  == "zag_fire":
                    name="zagfire{}".format(i+2)
                    to_add=ZagFire(in_channels= self.channel_counts[i], out_channels=e, proxy_ctx=proxy_ctx, proxy_mode=config.proxy_context_type,bypass_last=not config.zag_dont_bypass_last, activation=nn.ReLU(), dropout_rate=config.zag_fire_dropout)
                elif config.mode == "bnnfire":
                    name = "binaryfire{}".format(i+2)
                    bnn_pool_here= False
                    if config.bnn_pooling and (i+pool_offset) % config.pool_interval == 0 and i !=0:
                        bnn_pool_here=True
                    to_add = BNNFire(binarize_ctx = proxy_ctx ,in_channels= self.channel_counts[i], out_channels = e, pool= bnn_pool_here, use_prelu = config.bnn_prelu) 
                elif config.mode == "shuffle_fire":
                    name="shuffle_fire{}".format(i+2)
                    to_add = ShuffleFire(ShuffleFireSettings(in_chan = self.channel_counts[i], out_chan = e, groups1 = config.shuffle_fire_g1, groups2=config.shuffle_fire_g2, bottle_chan =num_squeeze   ,activation = F.leaky_relu, dont_wrap_sepconv =config.shuffle_fire_dont_wrap_sepconv ), proxy_ctx )  
                elif config.mode == "vgg_fire":
                    name="vgg_fire{}".format(i+2)
                    to_add = VGGFire(in_channels=self.channel_counts[i], out_channels = e,proxy_ctx=proxy_ctx, proxy_mode= config.proxy_context_type )
                else:
                    raise Exception("Unknown Fire")
                    name="fire{}".format(i+2)
                    to_add=Fire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))


                if config.use_excitation:
                    to_add.skip=False 
                    if config.excitation_shake_shake:
                        to_add2.skip=False
                    else:
                        to_add2=None
                    to_add=ExcitationFire(to_add, in_channels=self.channel_counts[i], out_channels=e, r=config.excitation_r, skip=skip_here, skipmode=config.skipmode, shake_shake_enable=config.excitation_shake_shake, fire_to_wrap2=to_add2, proxy_ctx=proxy_ctx, proxy_mode = config.proxy_context_type)
                    name="ExcitationFire{}".format(i+2)
                layer_dict[name]=to_add

                self.channel_counts.append(e)

            elif config.mode == "dense_fire": 
                    layer_dict["dense_fire{}".format(i+2)]=DenseFire(k0=self.channel_counts[i], num_subunits=config.dense_fire_depth_list[i], k=config.dense_fire_k, prop3=config.prop3 )
                    ts=max(math.floor(config.dense_fire_compression_level*config.dense_fire_k),1)
                    layer_dict["transition{}".format(i+2)]=nn.Conv2d(config.dense_fire_k, ts, 1)
                    self.channel_counts.append(ts)
            elif config.mode == "dense_fire_v2":
                layer_dict["dense_firev2{}".format(i+2)]=DenseFireV2Block(k0=self.channel_counts[i], num_subunits=config.dense_fire_depth_list[i], k=config.dense_fire_k, num_squeeze=4*config.dense_fire_k, dropout_rate= config.densenet_dropout_rate, bottleneck = not config.densenet_no_bottleneck)
                doutsize=self.channel_counts[i]+config.dense_fire_k*config.dense_fire_depth_list[i]
                ts=max(math.floor(config.dense_fire_compression_level*doutsize),1)
                layer_dict["transition{}".format(i+2)]=DenseFireV2Transition(doutsize, ts)
                self.channel_counts.append(ts)
            elif config.mode == "msd_fire":
                if i == 0:
                    layer_dict["msd_init_fire_{}".format(i+2)] =msdnet.MSDInitialFire(in_channels=self.channel_counts[i],out_channels_finest=config.msd_growth_rate,num_scales =config.msd_num_scales,proxy_ctx=proxy_ctx, proxy_ctx_mode = config.proxy_context_type    ) 
                    self.channel_counts.append(config.msd_growth_rate) #Note that the first column of a msdnet does not apped its input to its output, unlike the other columns
                else:
                    layer_dict["msd_column_fire_{}".format(i+2)] = msdnet.MSDColumnFire(in_channels_finest = self.channel_counts[i], growth_channels_finest =config.msd_growth_rate, num_scales = config.msd_num_scales, proxy_ctx= proxy_ctx, proxy_ctx_mode = config.proxy_context_type)
                    self.channel_counts.append(self.channel_counts[i] + config.msd_growth_rate)

            if not config.disable_pooling and (i+pool_offset) % config.pool_interval == 0 and (i !=0 or config.allow_pooling_after_first_fire) and i!=(num_fires -1):
                logging.info("adding max pool layer")
                layer_dict["maxpool{}".format(i+2)]= nn.MaxPool2d(kernel_size=config.max_pool_size,stride=2)
                num_pool_so_far+=1
        logging.info("counted " +str(num_pool_so_far)+" pooling layers" )
        logging.info("counted " +str(num_stride_so_far)+" strided layers" )
        skip_dropout = config.mode == "msd_fire"
        if not skip_dropout:
            layer_dict["dropout"]=nn.Dropout(p=config.dropout_rate)
        self.final_fc=config.final_fc
        assert( not self.final_fc) #temp bugged
        if not self.final_fc:
            if config.mode != "normal" and config.mode !="msd_fire":
              if config.final_mode=="normal":
                if config.final_act_mode == "enable":
                    layer_dict["final_convrelu"]=nn.LeakyReLU()
                if config.proxy_context_type == "no_context":
                    layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
                elif config.bypass_first_last:
                    layer_dict["final_conv"]=proxy_ctx.bypass(nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1)) 
                else:
                    layer_dict["final_conv"]=proxy_ctx.wrap(nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1)) 
              elif config.final_mode == "linear":
                  layer_dict["final_conv"] = FinalLinear(self.channel_counts[-1], config.final_side_length, config.final_side_length, config.out_dim, proxy_ctx=proxy_ctx)
            elif config.mode == "msd_fire":
                layer_dict["msd_final_exit"]= msdnet.MSDExitBranchFire( in_channels=self.channel_counts[-1],num_classes= config.out_dim,num_scales=config.msd_num_scales, proxy_ctx=proxy_ctx, proxy_ctx_mode=config.proxy_context_type)
            else: 
                layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
                if config.final_act_mode =="enable":
                    layer_dict["final_convrelu"]=nn.LeakyReLU()
        else:
            raise Exception("bugged")
            self.final_fc_weights=nn.Linear(self.channel_counts[-1], config.out_dim)

        if config.scale_layer:
            if config.proxy_context_type == "no_context":
                layer_dict["scale_layer"] = ScaleLayer()  
            else:
                layer_dict["scale_layer"] = proxy_ctx.bypass(ScaleLayer())
            


        self.layer_chunk_list=[]
        if config.use_non_default_layer_splits:
            layer_splits=config.layer_splits
        else:
            chunk_size = len(layer_dict.items()) // config.num_layer_chunks
            layer_splits=[]
            for i in range(config.num_layer_chunks):
                layer_splits.append(i*chunk_size)
            
        for i in range( config.num_layer_chunks -1 ):
            to_add_to_chunk=list(layer_dict.items())[layer_splits[i]:layer_splits[i+1] ]
            if config.use_forking and i in config.fork_after_chunks:
                if config.fork_module == "zag_fire":
                   last_layer_of_chunk = to_add_to_chunk[-1][1]
                   if isinstance(last_layer_of_chunk, nn.MaxPool2d):
                       last_layer_of_chunk = to_add_to_chunk[-2][1]

                   fork_module = ZagFire(in_channels =last_layer_of_chunk.out_channels, out_channels=config.out_dim, proxy_ctx=proxy_ctx, proxy_mode=config.proxy_context_type,bypass_last=not config.zag_dont_bypass_last, activation=nn.ReLU(), dropout_rate=config.zag_fire_dropout)
                fork_fire = ForkFire(fork_module=fork_module) 
                to_add_to_chunk.append(("fork", fork_fire))

            layer_chunk=nn.Sequential( collections.OrderedDict(to_add_to_chunk) )
            self.add_module("layer_chunk_"+str(i),layer_chunk )
            self.layer_chunk_list.append(layer_chunk)
        #add last chunk
        layer_chunk=nn.Sequential( collections.OrderedDict(list(layer_dict.items())[ layer_splits[config.num_layer_chunks-1]:]) )
        self.add_module("layer_chunk_"+str(config.num_layer_chunks-1),layer_chunk )
        self.layer_chunk_list.append(layer_chunk)
        #self.sequential=nn.Sequential(layer_dict)
        if config.proxy_context_type == "prune_context":
            assert len(proxy_ctx.list_proxies("weight_provider")) == len(proxy_ctx.list_proxies("weight_hook"))

    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by wifth
            reutrns:
                -oput is batchsize by config.outdim
        '''
        
        if self.use_forking:
            score_list=[]
            


        for i,layer_chunk in enumerate(self.layer_chunk_list):
            if self.chunk_across_devices:
                x=x.cuda(self.layer_chunk_devices[i])
            r=layer_chunk(x)
            if self.use_forking and i in self.fork_after_chunks:
                assert isinstance(r, tuple)
                cur_scores = r[1].mean(dim=3).mean(dim=2)
                if not self.training and self.fork_early_exit:
                    assert cur_scores.shape[0]==1 #cannot exit early with batches of size greater than 1
                    cur_scores_log_probs = F.log_softmax(cur_scores.view(-1))
                    cur_scores_probs= F.softmax(cur_scores.view(-1))
                    entropy = - torch.sum(cur_scores_probs*cur_scores_log_probs)
                    if entropy< self.fork_entropy_threshold:
                        if self.calculating_exit_proportions:
                            self.exit_tallies[i] += 1
                            self.total_exits+=1
                        return cur_scores
                    

                score_list.append(cur_scores)
                x=r[0]
            else:
                x=r

        x=torch.mean(x,dim=3)
        x=torch.mean(x,dim=2)
        #if self.final_fc:
        #    raise Exception("fc currently bugged")
        #    x=F.leaky_relu(self.final_fc_weights(x))
        if self.use_forking and self.training:
            score_list.append(x)
            if score_list[0].is_cuda: #move all scores to the same device
                for i in range(len(score_list)):
                    score_list[i]=score_list[i].cuda(score_list[0].get_device())
                return score_list
        if not self.training and self.fork_early_exit and self.calculating_exit_proportions:
            self.total_exits+=1
        return x


    def adjust_out_dim(self, new_out_dim, linear=False, spatial_dim=None):
        #replaces the final convolutional layer.  assumes we want to bypass it with the proxy ctx.
        #assert "final_conv" in self.layer_chunk_list[-1].keys()
        if linear:
            new_final=FinalLinear(self.channel_counts[-1],spatial_dim, spatial_dim, new_out_dim, self.proxy_ctx)
        else:
            new_final=self.proxy_ctx.bypass(nn.Conv2d(self.channel_counts[-1], new_out_dim, kernel_size=1))
            logging.info("new final conv has input channels {} and output channels {}".format(self.channel_counts[-1],new_out_dim))
        if next(self.layer_chunk_list[-1][-1].parameters()).is_cuda:
            new_final=new_final.cuda(next(self.layer_chunk_list[-1][-1].parameters()).get_device())


        self.layer_chunk_list[-1][-1] = new_final 
        #seems this is automatically registered
        







    def cuda(self,*params,**params2):
        if not self.chunk_across_devices:
            return  super().cuda(*params,**params2)
            
        for i,layer_chunk in enumerate(self.layer_chunk_list):
            layer_chunk.cuda( self.layer_chunk_devices[i] )
            logging.info("Chunk number "+ str(i)+" is on device number "+ str(next(layer_chunk.parameters()).get_device())  )
        if self.final_fc:
            self.final_fc_weights.cuda(self.layer_chunk_devices[-1])
        return self


    def train(self, mode=True):
        super().train(mode)
      #  if isinstance(self.proxy_ctx, candle.prune.GroupPruneContext) and self.proxy_ctx.stochastic and self.freeze_hard_concrete_for_testing:
      #      if mode == True:
      #          self.proxy_ctx.unfreeze()
      #      else:
      #          self.proxy_ctx.freeze()

    

    def init_params(self):
        init_p(self)

    def multiplies(self, img_h, img_w, input_channels,unpruned ):
         if self.use_forking:
             assert self.exit_proportions_calculated
             mults_by_chunk = []
             cur_h=img_h
             cur_w=img_w
             cur_channels=input_channels
             for chunk in self.layer_chunk_list:
                mults,cur_channels, cur_h,cur_w =count_approx_multiplies(chunk,cur_h, cur_w, cur_channels, unpruned= unpruned)
                mults_by_chunk.append(mults) 
             culm_mults_by_chunk = [mults_by_chunk[0]]
             for i in range(1,len(mults_by_chunk)):
                 culm_mults_by_chunk.append(culm_mults_by_chunk[i-1]+mults_by_chunk[i])
             avg_mults=0
             report_string=""
             for i in range(len(self.layer_chunk_list)):
                avg_mults+=culm_mults_by_chunk[i]*self.exit_proportions[i] 
                report_string +="+"+ str(culm_mults_by_chunk[i])+"*"+str(self.exit_proportions[i])
             report_string +="="+str(avg_mults)
             logging.info(report_string)
             return avg_mults

                          


         mults,_,_,_= count_approx_multiplies(self.layer_chunk_list,img_h=img_h, img_w=img_w,input_channels=input_channels,unpruned=unpruned ) 
         return mults

    def calc_exit_proportions(self):
       assert self.use_forking 
       assert self.calculating_exit_proportions
       logging.info("Total Exits:"+str(self.total_exits))
       logging.info("Exit tallies:"+",".join(map(str,self.exit_tallies)) )
       for index in range(self.num_layer_chunks):
            logging.info("chunk: "+str(index))
            logging.info("exit tally: "+str(self.exit_tallies[index]))
            self.exit_proportions[index] = self.exit_tallies[index]/self.total_exits
       self.exit_proportions[-1]=1-sum(self.exit_proportions[:-1])
       logging.info("exit proportions: "+str(self.exit_proportions))
       self.exit_proportions_calculated = True
       
    def prop_nonzero_masks(self):
        if self.mode == "next_fire": 
            prop1_list = []
            prop2_list = [] 
            prop3_list = []

            for chunk in self.layer_chunk_list:
                 for layer in chunk:
                     if isinstance(layer, NextFire):
                        p1, p2, p3 = layer.prop_nonzero_masks()
                        prop1_list.append(p1)
                        prop2_list.append(p2)
                        prop3_list.append(p3)

            return {"bn1" : prop1_list, "bn2" : prop2_list, "bn3" : prop3_list} 

    def prop_flop_reduction(self, img_h, img_w, input_channels ,unit_target = "by_mode"):
        if unit_target == "by_mode":
            if self.mode == "next_fire": 
                unit_target = NextFire
        props = [] 
        difs=[]
        cur_h_unpruned = img_h 
        cur_w_unpruned = img_w
        cur_channels_unpruned = input_channels
        cur_h_pruned = img_h
        cur_w_pruned = img_w
        cur_channels_pruned = input_channels
        for chunk in self.layer_chunk_list:
            for layer in chunk:
                pruned_mults, cur_channels_pruned, cur_h_pruned, cur_w_pruned = count_approx_multiplies(layer, img_h = cur_h_pruned,img_w = cur_w_pruned,input_channels=  cur_channels_pruned, unpruned = False)
                layer_mults, cur_channels_unpruned, cur_h_unpruned, cur_w_unpruned = count_approx_multiplies(layer, img_h = cur_h_unpruned,img_w = cur_w_unpruned, input_channels=  cur_channels_unpruned, unpruned = True)
                if isinstance(layer,unit_target):
                    props.append(pruned_mults / layer_mults)
                    difs.append(layer_mults - pruned_mults)
        return props, difs
    


    def reset_weights(self):
        for chunk in self.layer_chunk_list:
            for layer in chunk:
                reset_weights(layer)

    def reset_masks(self):
        for chunk in self.layer_chunk_list:
            reset_masks(chunk)

    def condense(self):
        def _condense(module):
            if getattr(layer,"condense",None) is not None:
                module.condense()
            try:
                for sublayer in module:
                    _condense(sublayer)
            except TypeError:
                return
        _condense(self.layer_chunk_list)

    def to_blocks(self):
        '''
         Converts a squeezenet into an OrderedDict
        '''
        blocks = collections.OrderedDict()
        for chunk in self.layer_chunk_list:
            for name, module in chunk.named_children():                
                blocks[name]=module
        return blocks

    def to_subblocks(self):
        '''
        as above, but if a module provides a method for breaking up into sublocks, use it
        '''
        sub_blocks = collections.OrderedDict()
        for chunk in self.layer_chunk_list:
            for name, module in chunk.named_children():                
                if getattr(module,"submods",None) is not None:
                    for subname, submod in module.submods():
                        sub_blocks[name +"_"+subname]=submod
                else:
                    sub_blocks[name]=module


        return sub_blocks



    def compute_pruning_normalization_factor(self, mode):
        for name, module in self.to_blocks().items():
            if getattr(module,"compute_pruning_normalization_factor",None) is not None:
                module.compute_pruning_normalization_factor(mode)


    def change_store_input(self,val):
        change_store_input(self.layer_chunk_list)
        
    def set_lambd_by_subblock(self,lambd_first,lambd_last):
        import candle.prune
        sub_list=list(filter(lambda x:isinstance(x, candle.prune.ProxyLayer),  self.to_subblocks().values()))
        num= len(sub_list)
        assert(num>1)
        a=(lambd_last/lambd_first)**(1/(num-1))
        for i,sub in enumerate(sub_list):
            assert(isinstance(sub.weight_provider,candle.prune.WeightMaskGroup))
            sub.weight_provider.local_l0_lambd=lambd_first*a**i

    def set_l0_lambda_by_name(self, name_dict):
        import candle.proxy
        sub_dict=self.to_subblocks()
        for name, sb in sub_dict.items():
            if isinstance(sb, candle.proxy.ProxyLayer):
                for key, val in name_dict.items():
                    if key in name:
                        sb.weight_provider.local_l0_lambd=val

    def display_subblock_nonzero_masks(self, warning=True,nodesc=False):
        import candle.prune
        sub_dict=self.to_subblocks()
        for name, sb in sub_dict.items():
            if not isinstance(sb, candle.prune.ProxyLayer):
                continue
            mask_len = sb.weight_provider._flattened_masks[0].size(0)
            mask_nonzero = sb.weight_provider.mask_unpruned[0]
            if nodesc:
                print("{},{},{}".format(name,mask_nonzero, mask_len))
            elif warning:
                logging.warning("name:{} Unpruned Masks: {} / {} ".format(name, mask_nonzero, mask_len))
            else:
                logging.info("name:{} Unpruned Masks: {} / {} ".format(name, mask_nonzero, mask_len))

    def compute_flop_reg_terms_standard(self, img_h, img_w, input_channels, mode="output_prune"):
        #assume all convolutions in the body of the neural network are proxy convs assoicated with output channel pruning
        self.multiplies( img_h, img_w, input_channels,unpruned=False ) #causes Proxy layers to record mults
        sub_dict=self.to_subblocks()
        proxy_conv_list=[]
        for sb in sub_dict.values():
            if isinstance(sb, candle.proxy.ProxyConv2d):
                proxy_conv_list.append(sb)
        
        for i,proxy_conv in enumerate(proxy_conv_list):
            flops_before=0
            flops_after=0
            flops_before+= proxy_conv.mults
            flops_after+= proxy_conv.mults_om1

            if  i< len(proxy_conv_list)-1:
                flops_before+= proxy_conv_list[i+1].mults
                flops_after += proxy_conv_list[i+1].mults_im1
            proxy_conv.flop_reg_term=flops_before - flops_after
        
    #compute the flop regularization terms for all proxy


    def deemph_string(self, string): #demphsizes for pruning proxy layers whose names contain  a given string
        import candle.proxy
        for name, layer in self.to_subblocks().items():
            if not isinstance(layer, candle.proxy._ProxyConvNd):
                continue
            if string in name:
                logging.info("deemphasizing {}".format(name))
                layer.deemph=True

       


            

def init_p(mod):
        '''
        Based on the initialization done int he pytorch resnet example https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L112-L118
        '''
        for m in mod.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.reset_parameters()

def forking_props_from_sample(squeezenet,  loader):
    assert squeezenet.use_forking
    assert not squeezenet.exit_proportions_calculated
    squeezenet.calculating_exit_proportions = True
    for batch, *other in loader:
        assert batch.shape[0] == 1 #to do early exit we need to have batch sizes of one
        squeezenet(batch) 
    squeezenet.calc_exit_proportions()

class FinalLinear(nn.Module):
    def __init__(self,in_c,in_h,in_w,out_c,proxy_ctx):
        super().__init__()
        self.in_dim=in_c*in_h*in_w
        self.out_c=out_c
        self.lin= proxy_ctx.bypass(nn.Linear(self.in_dim,out_c)) if proxy_ctx is not None else  nn.Linear(self.in_dim,out_c)
    def forward(self, x):
        x=x.view(-1, self.in_dim)
        x=self.lin(x)
        x=x.view(-1,self.out_c,1,1)
        return x
    def multiplies(self,img_h, img_w, input_channels, unpruned ):
        return input_channels*img_h*img_w*self.lin.weight.shape[1],self.out_c,1,1





class PruningNormalizationMode(Enum):
    BY_LAYER=0
    BY_BLOCK=1
    NO_NORMALIZATION=-1
