import logging
import collections
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
#Note that , when we refer to the in_channels or out_channels
# of a MSDblock, we are refering to the finest scale
#By column, I mean what the original paper calls a layer
#also note: the out channels of the initial fire are the actualy number of output channels (at the finest scale)
class MSDInitialFire(serialmodule.SerializableModule):
    def __init__(self, in_channels, out_channels_finest, num_scales, proxy_ctx, proxy_ctx_mode):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_finest = out_channels_finest
        out_channels= out_channels_finest
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
                conv_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, stride=2)),
                bn_wrapper(nn.BatchNorm2d(out_channels)),
                nn.ReLU() 
                ))
        self.first_column = first_column

    def forward(self, x):
        out = [None]*self.num_scales
        for s in range(self.num_scales):
            x = self.first_column[s](x)
            out[s] = x
        return out

    def multiplies(self,img_h, img_w, input_channels):
        '''
        when multiplies functions are applied to msd blocks, the input and output_channels may be lists
        '''
        logging.debug("computing multipies for initial column")
        mults,_ ,_,_ = countmult.count_approx_multiplies(self.first_column, img_h, img_w, input_channels)
        out_chan = []
        for i in range(self.num_scales):
            out_chan.append(self.out_channels_finest *2**i )
        return mults, out_chan, img_h, img_w #image dimensions at the finest scale are preserved   
            
class MSDConvUnit(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, proxy_ctx, proxy_ctx_mode):
        super().__init__()
        conv_wrapper, bn_wrapper = get_wrappers(proxy_ctx, proxy_ctx_mode) 
        self.add_module('conv1',conv_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), ))
        self.add_module('bn1',bn_wrapper(nn.BatchNorm2d(out_channels))  )
        self.add_module('relu1',nn.ReLU() ) 
        self.add_module('conv2',conv_wrapper(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=stride), ))
        self.add_module('bn2',bn_wrapper(nn.BatchNorm2d(out_channels))  )
        self.add_module('relu2', nn.ReLU())


class MSDColumnFire(serialmodule.SerializableModule):
    def __init__(self,in_channels_finest, growth_channels_finest, num_scales, proxy_ctx, proxy_ctx_mode):
        #recall the in_channels and growth channels refer only to the finest scale
       super().__init__() 
       self.num_scales = num_scales
       self.in_channels_finest = in_channels_finest
       self.growth_channels_finest = growth_channels_finest
       in_channels= in_channels_finest
       out_channels = growth_channels_finest
       diagonal_column = nn.ModuleList()
       horizontal_column = nn.ModuleList()
       horizontal_column.append(MSDConvUnit(in_channels=in_channels, out_channels = out_channels, stride=1, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode ))
       for s in range(1, num_scales):
           in_channels = 2*in_channels 
           out_channels = 2* out_channels
           horizontal_column.append(MSDConvUnit(in_channels= in_channels, out_channels = out_channels//2, stride =1, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode)  ) #horizonal and diagonal each responsible for half output channels
           diagonal_column.append(MSDConvUnit(in_channels= in_channels//2, out_channels = out_channels//2, stride =2, proxy_ctx=proxy_ctx, proxy_ctx_mode = proxy_ctx_mode)  )#diagonal gets input from previous scale
       self.horizontal_column = horizontal_column
       self.diagonal_column = diagonal_column

    def forward(self,x):
        out = [None] * self.num_scales 
        out[0] = torch.cat([x[0], self.horizontal_column[0](x[0])],dim=1)
        for s in range(1, self.num_scales):
            out_horizontal = self.horizontal_column[s](x[s])
            out_diagonal = self.diagonal_column[s-1](x[s-1])
            out[s] = torch.cat([x[s], out_horizontal, out_diagonal],dim=1)
        return out

    def multiplies(self,img_h, img_w, input_channels ):
        #note again that input_channels should be a list
        assert input_channels[0] == self.in_channels_finest
        mults = 0
        cur_h= img_h
        cur_w = img_w
        in_chan = self.in_channels_finest
        logging.debug("computing mutlipies for an MSDColumnFire")
        for s in range(self.num_scales):
            mults_h,_,_,_= countmult.count_approx_multiplies(self.horizontal_column[s],cur_h, cur_w, in_chan )
            if s<self.num_scales -1:
                mults_d,_,_,_= countmult.count_approx_multiplies(self.diagonal_column[s],cur_h, cur_w, in_chan )
            else:
                mults_d=0
            in_chan = 2 * in_chan
            cur_h= cur_h/2
            cur_w = cur_w/2
            this_scale_mults= mults_h +mults_d
            mults = mults + this_scale_mults 
            logging.debug("At scale {}, found {} mults from the horiziontal convolution, and {} mults from the outgoing diagonal convolution.  This yields a total of {} mults.".format(s,mults_h,mults_d,this_scale_mults))

        out_chan = []
        for i in range(self.num_scales):
            out_chan.append(self.growth_channels_finest *2**i +input_channels[i])
        return mults, out_chan, img_h, img_w
    def __repr__(self):
        s=super().__repr__()
        in_chan = []
        for i in range(self.num_scales):
            in_chan.append(self.in_channels_finest *2**i )
        growth_chan = []
        for i in range(self.num_scales):
            growth_chan.append(self.growth_channels_finest *2**i )
        out_chan=[]
        for i in range(self.num_scales):
            out_chan.append(growth_chan[i] +in_chan[i] )
           
        return self.__class__.__name__+"{} -> {} + {} = {} \n".format(in_chan,in_chan, growth_chan, out_chan) +s
       
    
class MSDExitBranchFire(serialmodule.SerializableModule):
    #uses the coursest scale of the output of an MSDColumnFire to get output suitable for classification
    def __init__(self, in_channels,num_classes,num_scales, proxy_ctx, proxy_ctx_mode):
        #again, here in_channels refers to cahnnels in the finest scale
        super().__init__()
        course_in_channels = in_channels*(2**(num_scales-1))
        conv_wrapper, bn_wrapper = get_wrappers(proxy_ctx, proxy_ctx_mode) 
        self.seq =nn.Sequential(
        collections.OrderedDict([
        ('conv1',conv_wrapper(nn.Conv2d(course_in_channels, num_classes, kernel_size=1) )),
        ('bn1',bn_wrapper(nn.BatchNorm2d(num_classes))  ),
        ('relu1',nn.ReLU() ),
        ('conv2',conv_wrapper(nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1) )),
        ('bn2',bn_wrapper(nn.BatchNorm2d(num_classes))  ),
        ('relu2', nn.ReLU())
        ]))
    def forward(self, x):
        return self.seq(x[-1])

    def multiplies(self,img_h, img_w, input_channels ):
        logging.debug("counting multiplies for MSDExitBranchFire")
        return countmult.count_approx_multiplies(self.seq, img_h, img_w, input_channels[-1])

