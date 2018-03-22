import math
import collections
import logging
import random
import copy
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from . import serialmodule
from . import shakedrop_func 
from .countmult import count_approx_multiplies
from . import shake_shake
from torch.autograd import Variable

from candle.prune import PruneContext, GroupPruneContext

import candle.context
import candle.prune
import candle.quantize
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
    parser.add_argument("--squeezenet_sr",type=float, default=0.125)
    parser.add_argument("--squeezenet_out_dim",type=int)

    parser.add_argument("--squeezenet_mode",type=str, choices=["shuffle_fire", "bnnfire", "resfire","wide_resfire","dense_fire","dense_fire_v2","next_fire","normal"], default="normal")

    parser.add_argument("--squeezenet_dropout_rate",type=float,default=0)
    parser.add_argument("--squeezenet_densenet_dropout_rate",type=float,default=0)

    parser.add_argument("--fire_skip_mode", type=str, choices=["simple", "none", "zero_pad"], default= "none")
    parser.add_argument("--squeezenet_pool_interval_mode",type=str, choices=["add","multiply"], default="add")
    parser.add_argument("--squeezenet_pool_interval",type=int, default=4)
    parser.add_argument("--squeezenet_num_fires", type=int, default=8)
    parser.add_argument("--squeezenet_conv1_stride", type=int, default=2)
    parser.add_argument("--squeezenet_conv1_size",type=int, default=7)
    parser.add_argument("--squeezenet_num_conv1_filters", type=int, default=96) 
    parser.add_argument("--squeezenet_pooling_count_offset", type=int, default=2) #should be greater than 0, otherwise you get a pool in the first layer
    parser.add_argument("--squeezenet_max_pool_size",type=int, default=3)
    parser.add_argument("--squeezenet_disable_pooling",action="store_true")

    parser.add_argument("--squeezenet_dense_k",type=int, default=12)
    parser.add_argument("--squeezenet_dense_fire_depths",type=str, default="default, shallow")
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

    
    parser.add_argument("--squeezenet_final_act_mode", choices=["enable", "disable"], default="enable", help="should there be an activation with the final conv")
    parser.add_argument("--squeezenet_scale_layer",action="store_true")

    parser.add_argument("--squeezenet_shuffle_fire_g1",default= 8)
    parser.add_argument("--squeezenet_shuffle_fire_g2",default= 8)

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

class NextFire(serialmodule.SerializableModule):

    def __init__(self, in_channels, num_squeeze, num_expand,skip,skipmode, groups=32, final_bn=False,stochastic_depth=False, survival_prob=1, shakedrop=False, shake_shake=False, shake_shake_mode= shake_shake.ShakeMode.IMAGE, proxy_ctx=None,proxy_mode = None):
        super().__init__()
        layer_dict = collections.OrderedDict()
        layer_dict["bn1"] = nn.BatchNorm2d(in_channels)
        layer_dict["leaky_reu1" ] = nn.LeakyReLU(inplace=True)
        layer_dict["squeeze_conv"] = nn.Conv2d(in_channels, num_squeeze, (1,1))
        layer_dict["bn2"] = nn.BatchNorm2d(num_squeeze)
        layer_dict["leaky_relu2"] = nn.LeakyReLU(inplace=True)
        layer_dict["group_conv"] = nn.Conv2d(num_squeeze, num_squeeze, kernel_size=3, padding=1, groups=groups)
        layer_dict["bn3"] = nn.BatchNorm2d(num_squeeze)
        layer_dict["leaky_relu3"] = nn.LeakyReLU(inplace=True)
        layer_dict["expand_conv"] =  nn.Conv2d(num_squeeze, num_expand, kernel_size=1)
        if final_bn:
            logging.info("Making NextFire layer with a final batchnorm")
            layer_dict["final_bn"]=nn.BatchNorm2d(num_expand)
        if proxy_mode == "prune_context" :
            logging.info("wrapping layers using a proxy context for standard pruning")
            layer_dict = proxy_ctx.wrap_dict(layer_dict)
        if proxy_mode == "group_prune_context":
            logging.info("wrapping layers for channel-wise pruning")
            assert(groups==1) #convolutional groupwise pruning not implemented yet
            layer_dict = proxy_ctx.wrap_dict(layer_dict)
        elif proxy_mode == "l0reg_context":
            logging.info("wrapping layers using a proxy context for l0 regularization  pruning")
            layer_dict = proxy_ctx.wrap_dict(layer_dict) 
        elif proxy_ctx is not None:
            raise Exception("unknown proxy_mode")
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

    def multiplies(img_h, img_w, effective_input_channels):
        assert not self.shake_shake and not self.shake_drop and not self.stochastic_depth
        mults, _,_,_ =  count_approx_multiplies(layer=self.seq, img_h=img_h, img_w=img_w)        
        return mults, effective_input_channels,img_h ,img_w #residual branch restores effective_input_channels
        

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
            self.pool_layer = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
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

    def multiplies(img_h, img_w, effective_input_channels):
        assert not self.shake_shake
        compress_mults, compress_out_dim = count_effective_mults(self.compress,  effective_input_dim = effective_input_channels )
        expand_mults, _ = count_effective_mults(self.expand,  effective_input_dim = compress_out_dim )
        wrapped_mults, _,out_h, out_w = count_effective_mults(self.wrapped,img_h=img_h, img_w=img_w, effective_input_channels=effective_input_channels)
        return compress_mults + wrapped_mults + expand_mults, effective_input_channels, out_h, out_w


class ScaleLayer(serialmodule.SerializableModule):
    def __init__(self,init_val=0.001):
        super().__init__()
        self.scalar = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x):
        return self.scalar*x

#observation: if groups are of size h in both first and final pointwise convolutions, then an output channel depends on h^2 input channels
ShuffleFireSettings = collections.namedtuple("ShuffleFireSettings","in_chan, bottle_chan, out_chan, groups1, groups2,activation")
class ShuffleFire(serialmodule.SerializableModule):

    def __init__(self, settings, ctx = None):
        super().__init__()
        assert(settings.in_chan % settings.groups1 == 0)
        assert(settings.bottle_chan % settings.groups1 == 0 )
        assert(settings.bottle_chan % settings.groups2 == 0 )
        assert(settings.out_chan % settings.groups2 == 0)
        wrap = ctx.wrap if ctx is not None else lambda w: w
        bypass = ctx.bypass if ctx is not None else lambda w: w

        self.bn1 = bypass( nn.BatchNorm2d(settings.in_chan))
        self.gconv1 = wrap( nn.Conv2d(in_channels=settings.in_chan, out_channels=settings.bottle_chan, kernel_size=1, groups=settings.groups1) )
        self.bn2 = bypass( nn.BatchNorm2d(settings.bottle_chan))
        self.sepconv = wrap(nn.Conv2d(in_channels=settings.bottle_chan, out_channels=settings.bottle_chan, kernel_size=3, padding=1, groups=settings.bottle_chan))
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
    '''
    def __init__(self, input_size, k, num_squeeze, dropout_rate ):
        super().__init__()
        layerdict = collections.OrderedDict()
        layerdict["batchnorm1"]=nn.BatchNorm2d(input_size)
        layerdict["relu1"]=nn.ReLU(inplace=True)
        layerdict["conv1"]=nn.Conv2d(input_size, num_squeeze, kernel_size=1)
        layerdict["batchnorm2"]=nn.BatchNorm2d(num_squeeze)
        layerdict["relu2"]=nn.ReLU(inplace=True)
        layerdict["conv2"]=nn.Conv2d(num_squeeze, k, kernel_size=3, padding=1)
        if dropout_rate>0:
            layerdict["droupout"]=nn.Dropout(p=dropout_rate)
        self.seq=nn.Sequential(layerdict)
    def forward(self, x):
        return torch.cat([x, self.seq(x)], dim = 1  )

        
class DenseFireV2Block(serialmodule.SerializableModule):
    def __init__(self, k0, k, num_subunits, num_squeeze, dropout_rate):
        super().__init__()
        layer_dict=collections.OrderedDict()
        cur_channels=k0
        for i in range(num_subunits):
            layer_dict["section{}".format(i)]=DenseFireV2Section(input_size=cur_channels, k=k, num_squeeze=num_squeeze, dropout_rate = dropout_rate )
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


SqueezeNetConfig=collections.namedtuple("SqueezeNetConfig","in_channels, base, incr, prop3, freq, sr, out_dim, skipmode,  dropout_rate, num_fires, pool_interval, conv1_stride, conv1_size, pooling_count_offset, num_conv1_filters,  dense_fire_k,  dense_fire_depth_list, dense_fire_compression_level, mode, use_excitation, excitation_r, pool_interval_mode, multiplicative_incr, local_dropout_rate, num_layer_chunks, chunk_across_devices, layer_chunk_devices, next_fire_groups, max_pool_size,densenet_dropout_rate, disable_pooling, next_fire_final_bn, next_fire_stochastic_depth, use_non_default_layer_splits, layer_splits, next_fire_shakedrop, final_fc, final_size, next_fire_shake_shake,excitation_shake_shake, proxy_context_type,bnn_pooling, final_act_mode, scale_layer,bnn_prelu, shuffle_fire_g1, shuffle_fire_g2")
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
                shuffle_fire_g2 = args.squeezenet_shuffle_fire_g2
                )
        return SqueezeNet(config)

    def __init__(self, config):
        super().__init__()
        self.chunk_across_devices=config.chunk_across_devices
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
        elif config.proxy_context_type == "no_context":
            proxy_ctx = None
        elif config.proxy_context_type == "tanhbinarize_context":
            proxy_ctx = candle.quantize.TanhBinarizeContext() 
        else:
            raise Exception("unknown proxy_context_type")

        self.proxy_ctx=proxy_ctx
        num_fires=config.num_fires #8
        first_layer_num_convs=config.num_conv1_filters
        first_layer_conv_width=config.conv1_size
        first_layer_padding= first_layer_conv_width // 2  

        pool_offset=config.pooling_count_offset

        if config.mode != "normal":
            layer_dict=collections.OrderedDict()
            if config.proxy_context_type == "no_context":
                layer_dict["conv1"] = nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)
            else:
                layer_dict["conv1"] =proxy_ctx.wrap( nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride))

             
        else:
            layer_dict=collections.OrderedDict([
            ("conv1", nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)),
            ("conv1relu", nn.LeakyReLU()),
            ("maxpool1", nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
            ]) 
        num_pool_so_far =0 #for debugging
        self.channel_counts=[ first_layer_num_convs]#initial number of channels entering ith fire layer (labeled i+2 to match paper)
        for i in range(num_fires):
            if  config.mode != "dense_fire" and config.mode != "dense_fire_v2":
                if config.pool_interval_mode == "add":
                    e = config.base+math.floor(config.incr*math.floor(i/config.freq))
                elif config.pool_interval_mode == "multiply":
                    e=config.base* (config.multiplicative_incr ** math.floor(i/config.freq)) 
                num_squeeze=max(math.floor(config.sr*e),1)
                num_expand3=max(math.floor(config.prop3*e),1)
                num_expand1=e-num_expand3
                if config.skipmode == FireSkipMode.SIMPLE and e == self.channel_counts[i]:
                    skip_here=True
                    logging.info("Making simple skip layer.")
                elif config.skipmode == FireSkipMode.PAD:
                    skip_here=True
                    if e == self.channel_counts[i]:
                        logging.info("Padding is enabled, but channel count has not changed.  Simple skipping will occur")
                    else:
                        logging.info("Making Padding skip layer")
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
                    to_add = NextFire(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand=e, skip=skip_here, groups=config.next_fire_groups, skipmode=config.skipmode, final_bn=config.next_fire_final_bn, stochastic_depth=config.next_fire_stochastic_depth, survival_prob = survival_prob, shakedrop=config.next_fire_shakedrop, shake_shake= config.next_fire_shake_shake, proxy_ctx=proxy_ctx, proxy_mode = config.proxy_context_type )
                    if config.excitation_shake_shake:
                        to_add2= NextFire(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand=e, skip=skip_here, groups=config.next_fire_groups, skipmode=config.skipmode, final_bn=config.next_fire_final_bn, stochastic_depth=config.next_fire_stochastic_depth, survival_prob = survival_prob, shakedrop=config.next_fire_shakedrop, shake_shake= config.next_fire_shake_shake, proxy_ctx=proxy_ctx, proxy_mode = config.proxy_context_type )
                elif config.mode == "bnnfire":
                    name = "binaryfire{}".format(i+2)
                    bnn_pool_here= False
                    if config.bnn_pooling and (i+pool_offset) % config.pool_interval == 0 and i !=0:
                        bnn_pool_here=True
                    to_add = BNNFire(binarize_ctx = proxy_ctx ,in_channels= self.channel_counts[i], out_channels = e, pool= bnn_pool_here, use_prelu = config.bnn_prelu) 
                elif config.mode == "shuffle_fire":
                    to_add = ShuffleFire(ShuffleFireSettings(in_chan = self.channel_counts[i], out_chan = e, groups1 = config.shuffle_fire_g1, groups2=config.shuffle_fire_g2, bottle_chan =num_squeeze   ,activation = F.leaky_relu ), proxy_ctx )  
                else:
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
                layer_dict["dense_firev2{}".format(i+2)]=DenseFireV2Block(k0=self.channel_counts[i], num_subunits=config.dense_fire_depth_list[i], k=config.dense_fire_k, num_squeeze=4*config.dense_fire_k, dropout_rate= config.densenet_dropout_rate)
                doutsize=self.channel_counts[i]+config.dense_fire_k*config.dense_fire_depth_list[i]
                ts=max(math.floor(config.dense_fire_compression_level*doutsize),1)
                layer_dict["transition{}".format(i+2)]=DenseFireV2Transition(doutsize, ts)
                self.channel_counts.append(ts)
                

            if not config.disable_pooling and (i+pool_offset) % config.pool_interval == 0 and i !=0:
                logging.info("adding max pool layer")
                layer_dict["maxpool{}".format(i+2)]= nn.MaxPool2d(kernel_size=config.max_pool_size,stride=2,padding=1)
                num_pool_so_far+=1
        logging.info("counted " +str(num_pool_so_far)+" pooling layers" )
        layer_dict["dropout"]=nn.Dropout(p=config.dropout_rate)
        self.final_fc=config.final_fc
        assert( not self.final_fc) #temp bugged
        if not self.final_fc:
            if config.mode != "normal":
                if config.final_act_mode == "enable":
                    layer_dict["final_convrelu"]=nn.LeakyReLU()
                if config.proxy_context_type == "no_context":
                    layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
                else:
                    layer_dict["final_conv"]=proxy_ctx.wrap(nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1)) 
            else: 
                layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
                if config.final_act_mode =="enable":
                    layer_dict["final_convrelu"]=nn.LeakyReLU()
        else:
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
            layer_chunk=nn.Sequential( collections.OrderedDict(list(layer_dict.items())[layer_splits[i]:layer_splits[i+1] ]) )
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
        



        for i,layer_chunk in enumerate(self.layer_chunk_list):
            if self.chunk_across_devices:
                x=x.cuda(self.layer_chunk_devices[i])
            x=layer_chunk(x)

        x=torch.mean(x,dim=3)
        x=torch.mean(x,dim=2)
        if self.final_fc:
            x=F.leaky_relu(self.final_fc_weights(x))
        return x


    def cuda(self,*params,**params2):
        if not self.chunk_across_devices:
            return  super().cuda(*params,**params2)
            
        for i,layer_chunk in enumerate(self.layer_chunk_list):
            layer_chunk.cuda( self.layer_chunk_devices[i] )
            logging.info("Chunk number "+ str(i)+" is on device number "+ str(next(layer_chunk.parameters()).get_device())  )
        if self.final_fc:
            self.final_fc_weights.cuda(self.layer_chunk_devices[-1])
        return self
    

    def init_params(self):
        init_p(self)

    def multiplies(self, img_h, img_w, input_channels ):


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


