import logging
import torch.nn as nn 
import math
def count_approx_multiplies(layer,img_h,img_w, input_channels):
    '''
    img_h: height of image
    img_w: width of image
    input_channels: The effective number of input channels that layer is to be fed, taking into account any zeroing of output channels that may have occured as the result of pruning in previous layers
    returns:
    -1.  count of approximate multiplies carried out by this channel
    -2.  Count of effective output channels produced by this channel
    -3. height of output image
    -4 width of output image
    '''
    logging.debug("count_approx_multiplies called on "+layer.__class__.__name__ +"  with img_h:"+str(img_h)+" img_w"+str(img_w)+" input_channels:"+str(input_channels)   )
    if isinstance(layer, nn.BatchNorm2d):
        logging.debug("returning 0 multiplies")
        return 0, input_channels, img_h, img_w
    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
        logging.debug("returning 0 multiplies")
        return 0, input_channels, img_h, img_w
    if isinstance(layer, nn.Dropout):
        logging.debug("returning 0 multiplies")
        return 0,input_channels, img_h,img_w
    if isinstance(layer, nn.MaxPool2d):
        padding = layer.padding
        if isinstance(padding,int):
            padding= (padding, padding)
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, int ):
            kernel_size= (kernel_size, kernel_size)
        stride = layer.stride
        if isinstance(stride, int):
            stride= (stride, stride)
        dilation = layer.dilation
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        out_h = math.floor((img_h +2*padding[0] -dilation[0]*(kernel_size[0] - 1  ) - 1)/stride[0]  )
        out_w = math.floor((img_w +2*padding[1] -dilation[1]*(kernel_size[1] - 1  ) - 1)/stride[1]  )
        logging.debug("returning 0 multiplies.  Changing image dimension to "+str(out_h)+" by "+str(out_w))
        return 0, input_channels, out_h, out_w 
    if isinstance(layer, nn.Conv2d):
        dim=layer.weight.data.shape
        mults=dim[0]*dim[1]*dim[2]*dim[3]*img_h*img_w
        logging.debug("found bypassed conv layer.  Reporting "+str(mults)+ " mults")
        return mults, layer.weight[0], img_h, img_w

        

    #see if layer implements a multiplies method
    if getattr(layer,"multiplies",None) is not None:
        return layer.multiplies(img_h=img_h, img_w=img_w, input_channels = input_channels)
    #see if layer is iterable
    total=0
    sublayer_channels = input_channels
    sublayer_h=img_h
    sublayer_w=img_w
    for sublayer in layer:
            sublayer_mult, sublayer_channels, sublayer_h, sublayer_w = count_approx_multiplies(sublayer, img_h = sublayer_h,img_w = sublayer_w,input_channels=  sublayer_channels)
            try:
                total += sublayer_mult
            except:
                import pdb; pdb.set_trace()
    return total, sublayer_channels, sublayer_h, sublayer_w
    
   # raise TypeError("Unable to compute multiplies for layer of type " + layer.__class__.__name__)
