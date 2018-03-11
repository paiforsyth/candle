import torch.nn as nn
from . import serialmodule
class SaveableDataPar(serialmodule.SerializableModule):
    def __init__(self,wrapped, device_ids):
        super().__init__()
        import pdb; pdb.set_trace()
        self.net= nn.DataParallel(wrapped,device_ids= device_ids)
        self.wrapped_list = [wrapped]
    def forward(self,x):
        return self.net(x)
    def init_params():
        self.wrapped_list[0].init_params
