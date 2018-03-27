import math
import torch
import torch.nn.functional as F

from . import proxy
from . import context
from . import nested
#big mistakes below.  Need to chuck output images allong spatial dimensions
class StdFactorizeConv2d(proxy.ProxyLayer):
    def __init__(self, weight_provider , stride=1, padding=0, dilation=1, groups=1, **kwargs):
        super().__init__(weight_provider, **kwargs)
        sizes = weight_provider.sizes.reify()
       
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
        self.factorize=False
        self.factorize_mode=None
        self.P_weights=None
        self.W_prime_weights=None
        self.factorized_bias=None 

        #save samples from forward pass for use in factorization
        self.save_samples=False
        self.saved_samples_mat = torch.Tensor(sizes[0][0],1)
    def cuda(self ):
        super().cuda()
        self.saved_samples_mat=self.saved_samples_mat.cuda()


    def on_forward(self, x):
        if self.factorize: 
           assert not self.training 
           return F.conv2d(F.conv2d(x, self.W_prime_weights,bias=None,**self.__conv__kwargs), self.P_weights, bias=self.factorized_bias )
        else:
            weights = self.weight_provider().reify()
            y= self.conv_fn(x, *weights, **self._conv_kwargs)
            if self.save_samples:
                imprt pdb; pdb.set_trace()
                self.saved_samples_mat=torch.cat([self.saved_samples_mat, 
                    y.transpose(1,0).contiguous().view(y.shape[1],-1).data 
                    ],dim=1)
            return y

    def multiplies(self,img_h, img_w, input_channels):
        w_dim = self.weight_provider.sizes.reify()[0]
        assert w_dim[1] == input_channels
        effective_out = w_dim[0]
        if not self.factorize:
           mults = img_h*img_w*w_dim[2]*w_dim[3] * input_channels *effective_out
        elif self.factorize_mode == "svd":
           rank = P_weights.shape[1]
           mults =  img_h*img_w*rank*effective_out + img_h*img_w*rank*w_dim[2]*w_dim[3]*input_channels
       
        return mults, effective_out, img_h, img_w 



    def do_svd_factorize(self,  sample_y=None, **kwargs ):
        '''
        Sample y should be a package of 1 by 1 sections of the output produced by this layer.  If it is None, will try use the saved_samples
        '''
        assert not self.training
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
        import pdb; pdb.set_trace()
        y_mean = samples_y.mean(dim = 1) 
        Y = sample_y - y_mean
        U,_,_ = torch.svd(Y)
        U=U[:,:target_rank] #truncate
        self.W_prime_weights = (U.transpose(1,0).mm(w_mat)).view(target_rank,w_dim[1], w_dim[2], w_dim[3])
        self.P_weights = Varaible(U.view(w_dim[0], target_rank, 1, 1) )
        self.factorized_bias = U.mm( U.transpose(1,0)).mm(w_bias) + (y_mean-U.mm(U.transpose(1,0)).mm(y_mean)) 

        self.saved_samples_mat = self.saved_samples_mat.new(self.saves_samples_mat.shape[0],1) 


class StdFactorizeContext(context.Context):
    def __init__(self, config=None, **kwargs):
        super().__init__(config,**kwargs)
        self.proxy_layers=[]
    def compose(self, layer, **kwargs):
        proxy_layer= super().compose(layer,factorize_method ="std") 
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
            prox_layer.saved_samples_list=[]



#wrong an unnecesary
def extract_kbyk_list(img,k):
   '''
   Given a channels by k by k img
   return a list consisting of vectors, each of which
   is reshaped from a k by k spatial section of the image
   '''

   vectorlist=[]
   channels = img.shape[0]
   img_h = img.shape[1]
   img_w = img.shape[2]
   padding_size = max(math.floor(k/2),1)
   kernel_extent = math.floor(k/2)
   assert kernel_extent < k/2 #assume odd k
   padding_h = img.new(channels, padding_size, img_w).fill_(0)
   padding_w = img.new(channels,2*padding_size+ img_h, padding_size).fill_(0)

   padded_img = torch.cat([padding_h, img, padding_h  ], dim=1 )
   padded_img = torch.cat([padding_w, padded_img, padding_w  ], dim=2  )
   for i in range(padding_size, padding_size+img_h):
       for j in range(padding_size, padding_size+img_w):
           vectorlist.append( (padded_img[:,i-kernel_extent:i+kernel_extent+1, j-kernel_extent:j+kernel_extent+1  ]).contiguous().view(-1,1)  ) #+1 neccesary since python does not include the endpoints of slices
   return vectorlist

