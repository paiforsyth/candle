import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from . import proxy
from . import context
from . import nested
class StdFactorizeConv2d(proxy.ProxyLayer):
    def __init__(self, weight_provider, svd_rank, use_factors, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        super().__init__(weight_provider, **kwargs)
        sizes = weight_provider.sizes.reify()
        wsize= sizes[1] 
        self.bias = len(sizes) == 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        assert groups ==1 #groups not yet implemented
        self.groups = groups
        self.conv_fn =  F.conv2d
        self._conv_kwargs = dict(dilation=dilation, padding=padding, stride=stride,groups=groups)
        if not self.bias:
            self._conv_kwargs["bias"] = None
        #for factorization
        self.factorize=use_factors
        self.factorize_mode=None
        self.W_prime_weights= Parameter(torch.Tensor(svd_rank,wsize[1],wsize[2],wsize[3]   )) #dummy
        self.P_weights= Parameter(torch.Tensor(wsize[0],svd_rank,wsize[1],wsize[3])) #dummy
        self.factorized_bias= Parameter(torch.Tensor(wsize[0]  )) #dummy

        #save samples from forward pass for use in factorization
        self.save_samples=False
        self.saved_samples_mat = None



    def on_forward(self, x):
        if self.factorize: 
           return F.conv2d(F.conv2d(x, self.W_prime_weights,bias=None,**self.__conv__kwargs), self.P_weights, bias=self.factorized_bias )
        else:
            weights = self.weight_provider().reify()
            y= self.conv_fn(x, *weights, **self._conv_kwargs)
            if self.save_samples:
                y_reshaped = y.transpose(1,0).contiguous().view(y.shape[1],-1).data 
                self.saved_samples_mat = y_reshaped if self.saved_samples_mat is None else torch.cat([self.saved_samples_mat,y_reshaped],dim=1)
            return y

    def multiplies(self,img_h, img_w, input_channels):
        w_dim = self.weight_provider.sizes.reify()[0]
        assert w_dim[1] == input_channels
        effective_out = w_dim[0]
        if not self.factorize:
           mults = img_h*img_w*w_dim[2]*w_dim[3] * input_channels *effective_out
        elif self.factorize_mode == "svd":
           rank = self.P_weights.shape[1]
           mults =  img_h*img_w*rank*effective_out + img_h*img_w*rank*w_dim[2]*w_dim[3]*input_channels
       
        return mults, effective_out, img_h, img_w 



    def do_svd_factorize(self,  sample_y=None, **kwargs ):
        '''
        Sample y should be a package of 1 by 1 sections of the output produced by this layer.  If it is None, will try use the saved_samples
        '''
        assert not self.training
        assert not self.factorize
        self.factorize_mode="svd"
        w_dim = self.weight_provider.sizes.reify()[0]

        self.factorize=True
        weight_list = self.weight_provider().reify()
        w_mat = weight_list[0].view(w_dim[0],-1).data
        w_bias = weight_list[1].data


        if "target_rank" in kwargs.keys():
            target_rank = kwargs["target_rank"]
        elif "rank_prop" in kwargs.keys():
            target_rank =math.ceil( kwargs["rank_prop"] * w_dim[0])
        else:
            raise Exception("No target rank information")

       
        if sample_y is None:
           sample_y = self.saved_samples_mat
        assert(sample_y.shape[0] < sample_y.shape[1])
        y_mean = sample_y.mean(dim = 1) 
        Y = sample_y - y_mean.view(-1,1)
        U,_,_ = torch.svd(Y.mm(Y.transpose(1,0)))
        U=U[:,:target_rank] #truncate
        W_prime_weights = (U.transpose(1,0).mm(w_mat)).view(target_rank,w_dim[1], w_dim[2], w_dim[3])
        P_weights = U.view(w_dim[0], target_rank, 1, 1) 
        M=U.mm(U.transpose(1,0))
        factorized_bias = M.mv(w_bias) + y_mean - M.mv(y_mean)  
        self.W_prime_weights.data = W_prime_weights
        self.P_weights.data = P_weights
        self.factorized_bias.data = factorized_bias

        self.saved_samples_mat = None 


class StdFactorizeContext(context.Context):
    def __init__(self, config=None, **kwargs):
        super().__init__(config,**kwargs)
        self.proxy_layers=[]

    def compose(self, layer, **kwargs):

        proxy_layer= super().compose(layer,factorize_method ="std",**kwargs) 
        self.proxy_layers.append(proxy_layer)
        return proxy_layer

    def list_model_params(self):
        return super().list_params()

    def factorize_all(self, strategy, **kwargs):
        for proxy_layer in self.proxy_layers:
            if strategy == "svd":
                proxy_layer.do_svd_factorize(rank_prop = kwargs["rank_prop"] )

    def save_samples_all(self,enable=True):
        for proxy_layer in self.proxy_layers:
            proxy_layer.save_samples=enable

    def clear_samples_all(self):
        for proxy_layer in self.proxy_layers:
            proxy_layer.save_samples=False
            proxy_layer.saved_samples_mat=None



