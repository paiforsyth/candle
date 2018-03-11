from torch.autograd import Variable, Function
from enum import Enum 

class ShakeMode(Enum):
    IMAGE=0
    BATCH=1

class ShakeFunc(Function):
    '''
    Modified from the implementation of hysest on Github
    '''
    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.alpha=alpha
        ctx.beta=beta
        return alpha*x1 +(1-alpha)*x2

    @staticmethod
    def backward(ctx, grad_output):
        alpha=ctx.alpha
        beta=Variable(ctx.beta)
        grad_x1 = grad_x2 = grad_alpha =grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = beta * grad_output 
        if ctx.needs_input_grad[1]:
            grad_x2 = (1-beta)* grad_output 
        return grad_x1, grad_x2, grad_alpha, grad_beta

    
def generate_alpha_beta(x, shake_mode, training):
        '''
        Note x is only used for its device and its size
        poissvlw forward  modes = 
        '''
        batch_size=x.shape[0]
        if training:
            if shake_mode == ShakeMode.IMAGE:
                alpha = x.data.new(batch_size,1,1,1).uniform_()
                beta = x.data.new(batch_size,1,1,1).uniform_()
            elif shake_mode == ShakeMode.BATCH:
                alpha = x.data.new(1).uniform_()
                beta = x.data.new(1).uniform_()
        else:
            alpha = x.data.new(1).fill_(0.5)
            beta= x.data.new(1).fill(0.5)
        return alpha, beta
        

       

