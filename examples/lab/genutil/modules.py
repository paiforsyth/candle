import numpy as np
#from pytorch forums
def count_trainable_params(model):   
   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
   param_count = count_elem(model_parameters)  
   return param_count

def count_elem(tensor_list):
    return sum( np.prod(p.size()) for p in tensor_list )


def get_named_trainable_param_tensors(model):
   named_model_parameters = filter(lambda p: p[1].requires_grad, model.named_parameters())
   return [ (p[0],p[1].data.clone()) for p in named_model_parameters ]
   


