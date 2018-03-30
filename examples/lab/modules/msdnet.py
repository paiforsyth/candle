import torch
import torch.nn as nn
from . import serialmodule
from . import countmult
def get_wrappers(proxy_ctx, proxy_ctx_mode):
        if proxy_ctx_mode == "no_context":
            conv_wrapper = bn_wrapper = lambda x:x 
        else:
            conv_wrapper = proxy_ctx.wrap
            bn_wrapper = proxy_ctx.bypass 
        return conv_wrapper, bn_wrapper


#reimplementation of MSDNet of HUANG et al., which was impelmented in pytorch bycontinue7777 on github
#Note that in everythinh below, when we refer to the in_channels or out_channels
# of a MSDblock, we are refering to the finest scale
#By column, I mean what the original paper calls a layer
class MSDInitialFire(serialmodule.SerializableModule):
    def __init__(self, in_channels, out_channels, num_scales, proxy_ctx,proxy_ctx_mode):
        super().init()
        self.input_channels = input_channels
        self.out_channels = output_channels
        self.num_scales=num_scales
        conv_wrapper, bn_wrapper = get_wrappers(proxy_ctx, proxy_ctx_mode) 
        first_column = nn.ModuleList()
        first_column.append(
                    nn.Sequential(
                    conv_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                    bn_wrapper(nn.BatchNorm2d(out_channels)),
                    nn.ReLU()
                    ))
        for s in range(1, num_scales):
            in_channels = out_channels
            out_channels = 2*out_channels
            first_column.append(nn.Sequential(
                conv_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, stride=2))
                bn_wrapper(nn.BatchNorm2d(out_channels))
                nn.ReLU() 
                ))
        self.first_column = first_column

     def forward(self, x):
        out = [None]*self.num_scales
        for s in range(self.num_scales):
            x = self.first_column[s](x)
            out[s] = x
        return out

    def multiplies(img_h, img_w, input_channels):
        '''
        Note: in  'multiplies' functions of MSD blocks, the 'channels' field refers to only channels of the finest scale
        '''
        mults,_ ,_,_ = countmult.count_approx_multiplies(self.first_column, img_h, img_w, input_channels)
        return mults, self.output_channels, img_h, img_w #image dimensions at the finest scale are preserved   
            
class MSDConvUnit(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, proxy_ctx, proxy_ctx_mode):
        conv_wrapper, bn_wrapper = get_wrappers(proxy_ctx, proxy_ctx_mode) 
        self.add_module('conv1',conv_wrapper(nn.conv2d(in_channels, out_channels, kernel_size=1, stride=stride), ))
        self.add_module('bn1',bn_wrapper(nn.batchnorm2d(out_channels))  )
        self.add_mofule('relu1', conv_wrapper(nn.conv2d(out_channels, out_channels ) ))
        self.add_module('conv2',conv_wrapper(nn.conv2d(out_channels, out__channels, kernel_size=3, padding=1), ))
        self.add_module('bn1',bn_wrapper(nn.batchnorm2d(out_channels))  )
        self.add_mofule('relu1', conv_wrapper(nn.conv2d(out_channels, out_channels ) ))


class MSDColumnFire(serialmodule.SerializableModule):
    def __init__(self,in_channels, out_channels, num_scales, proxy_ctx, proxy_ctx_mode):
       super().init() 
       self.num_scales = num_scales
       self.input_channels = input_channels
       self.output_channels = output_channels
       diagonal_column = nn.ModuleList()
       horizontal_column = nn.ModuleList()
       horizontal_column.append(MSDConvUnit(in_channels=in_channels, out_channels = out_channels, stride=1, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode ))
       for s in range(1, num_scales):
           in_channels = 2*in_channels 
           out_channels = 2* out_channels
           horizontal_column.append(MSDConvUnit(in_channels= in_channels, out_channels = out_channels/2, stride =1, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode)  ) #horizonal and diagonal each responsible for half output channels
           diagonal_column.append(MSDConvUnit(in_channels= in_channels/2, out_channels = out_channels/2, stride =2, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode)  )#diagonal gets input from previous scale
        self.horizontal_column = horizontal_column
        self.diagonal_column = diagonal_column

    def forward(self,x):
        out = [None] * self.num_scales 
        out[0] = self.horizontal_column[0](x[0])
        for s in range(1, self.num_scales):
            out_horizontal = self.horizontal_column[s](x[s])
            out_diagonal = self.diagonal_column[s-1](x[s-1])
            out[s] = torch.cat([x[s], out_horizontal, out_diagonal],dim=1)
        return out

    def multiplies(img_h, img_w, input_channels ):
        mults = 0
        cur_h= img_h
        cur_w = img_w
        in_chan = self.input_channels
        for s in range(self.num_scales):
            mults_h,_,_,_= horizontal_column[s].multiplies(cur_h, cur_w, in_chan )
            if s<self.num_scales -1:
                mults_d,_,_,_= diagonal_column[s].multiplies(cur_h, cur_w, in_chan )
            in_chan = 2 * in_chan
            cur_h= cur_h/2
            cur_w = cur_w/2
            mults = mults + mults_h + mults_d
        return mults, self.out_channels, img_h, img_w
    
    
def MSDSideBranchFire(nn.Sequential):
    #uses the coursest scale of the output of an MSDColumnFire to get output suitable for classification
    def __init__(self, num_channels,num_scales, proxy_ctx, proxy_ctx_mode):
        in_channels = num_channels/(2**num_scales)
        out_channels = in_channels
        conv_wrapper, bn_wrapper = get_wrappers(proxy_ctx, proxy_ctx_mode) 
        self.add_module('conv1',conv_wrapper(nn.conv2d(in_channels, out_channels, kernel_size=1) ))
        self.add_module('bn1',bn_wrapper(nn.batchnorm2d(out_channels))  )
        self.add_mofule('relu1', conv_wrapper(nn.conv2d(out_channels, out_channels ) ))
        self.add_module('conv2',conv_wrapper(nn.conv2d(out_channels, out__channels, kernel_size=3, padding=1), ))
        self.add_module('bn1',bn_wrapper(nn.batchnorm2d(out_channels))  )
        self.add_mofule('relu1', conv_wrapper(nn.conv2d(out_channels, out_channels ) ))


