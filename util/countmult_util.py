import math
import logging
def conv2d_mult_compute(img_h, img_w, in_channels, out_channels, groups, stride, padding, kernel_size, dilation):
        if isinstance(padding,int):
            padding= (padding, padding)
        if isinstance(kernel_size, int ):
            kernel_size= (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride= (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation) 
        if stride[0]!= 1:
            pass
            #import pdb; pdb.set_trace()
        out_h = math.floor((img_h +2*padding[0] -dilation[0]*(kernel_size[0] - 1  ) - 1)/stride[0] +1 )
        out_w = math.floor((img_w +2*padding[1] -dilation[1]*(kernel_size[1] - 1  ) - 1)/stride[1] +1 )
        logging.debug("counting multiplies for a 2d comvolution")
        logging.debug("Imput dimensions are  ({},{}) and output dimensions are ({},{}))".format(img_h,img_w,out_h,out_w ))
        if groups> 1:
            pass
            #import pdb; pdb.set_trace()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        mults = out_h * out_w *in_channels/groups * kernel_size[0] *kernel_size[1]*out_channels 
        logging.debug("number of multiplies is  {}*{}*{} / {} *{} * {} *{} = {} ".format(out_h,out_w,in_channels,groups,kernel_size[0], kernel_size[1],out_channels, mults ))
        logging.debug("number of output channels is "+ str(out_channels))
        return mults, out_channels, out_h, out_w

