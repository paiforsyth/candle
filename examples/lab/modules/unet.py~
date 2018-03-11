import torch
import torch.nn as nn
import torch.nn.functional as F
from . import serialmodule
import collections



BasicBlockConfig=collections.namedtuple("BasicBlockConfig", "in_channels, filters")
class BasicBlock(serialmodule.SerializableModule):
    def __init__(self,config):
        super().__init__()
        self.conv1=nn.Conv2d( config.in_channels, config.filters, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(config.fiters, config.filters, kernel_size=3, padding=1)

    def forward(self, x):
       out=F.leaky_relu(x) 
       out=self.conv1(out)
       out=F.leaky_relu(out)
       out=F.conv2(out)
       return out




UNetConfig= collections.namedtuple("UNetConfig", "in_channels, first_conv_filters,  out_classes, depth, blocktype")
class UNet(serialmodule.SerializableModule):
    def __init__(self, config):
      depth = config.depth
      self.first_conv=nn.conv2D(config.in_channels, config.first_conv_filters, kernel_size=3, padding=1)
      prev_filters=config.first_conv_Filters
      self.downlist=nn.ModuleList()
      self.size_list=[]
      for i in range(depth):
          cur_fliters = 2 * prev_filters
          self.size_list.append(cur_filters)
          if config.blocktype == "basic":
            downlist.append(BasicBlock(BasicBlockConfig(in_channels=prev_filters, filters=cur_filters) ))
          else:
              raise Exception("Unknown Block type")
          prev_filters = cur_filters
      

      self.uplist_across=nn.ModuleList()
      self.uplist_up=nn.Modulelist()
      for i in range(depthi-1):
          self.uplist_up.append( nn.ConvTranspose2d(self.size_list(depth-i),self.size_list(depth-i-1), kernel_size=2, stride=2))
          if blocktype == "bacic":
            self.uplist_across.append( basicBlock(BasicBlockConfig(in_channels = 2*self.size_list(depth-i-1) , filters=self.size_list(depth-i-1)   ) ) )

      self.final_conv=conv2D(self.size_list(0),config.out_classes)

    def forward(self, x):
        out = self.first_conv(x)
        down_outputs_list = []
        for i in range(depth):
            out=self.downlist[i](out)
            down_outputs_list.append(out)
            out = F.max_pool2d(out,kernel_size=2, stride=2)

        for i in range(depth):
            out=self.uplist_up[i](out)
            out=self.uplist_acorss[i](torch.cat([out, down_outputs_list[depth] ])  )

        out=self.final_conv(out)
        return out
