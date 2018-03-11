import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SerializableModule(nn.Module):
        def __init__(self):
            super().__init__()

        def save(self, filename):
            torch.save(self.state_dict(), filename)

        def load(self, filename):
            self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

ShuffleNetUnitSettings = collections.NamedTuple("ShuffleNetUnitSettings","in_chan, bottle_chan, out_chan, groups1, groups2")
class ShuffleNetUnit(SerializableModule):
    def __init__(self, ctx, settings):
        super().__init__()
        assert(settings.in_chan % settings.groups1 == 0)
        assert(settings.bottle_chan == settungs.groups1*settings.groups2 )
        assert(settings.out_chan % settings.groups2 == 0)
        self.gconv1 = ctx.wrap( nn.Conv2d(in_channels=settings.in_chan, out_channels=settings.bottle_chan, kernel_size=1, groups=settings.groups1) )
        self.bn1 = ctx.bypass( nn.BatchNorm2d(settings.bottle_chan))
        self.sepconv = ctx.wrap(nn.Conv2d(in_channels=settings.bottle_chan, out_channels=settings.bottle_chan, kernel_size=3, padding=0, groups=settings.bottle_chan))
        self.bn2 = ctx.bypass( nn.BatchNorm2d(settings.bottle_chan))
        self.gconv2 = ctx.wrap(nn.Conv2d(in_channels=settings.bottle_chan, out_channels=settings.out_chan, kernel_size=1, groups=settings.groups2))
        self.bn3 = ctx.bypass(nn.BatchNorm2d(settings.out_chan))
        
        self.groups1 = settings.groups1
        self.groups2 = settings.groups2
        self.in_chan = settings.in_chan
        self.out_chan = settings.out_chan

    def forward(self, x):
        shape = x.shape
        out = self.gconv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = out.view(shape[0],self.groups1,self.groups2,shape[2],shape[3])
        out = out.transpose(out, 2, 1)
        out = out.view(shape[0],self.groups2*self.groups1,shape[2],shape[3])

        out = self.sepconv(out)
        out = out.bn2(out)
        out = out.gconv2(out)
        out = out.bn3(out)
        if self.in_chan == self.out_chan:
            out=out+x
        else:
            padding = Variable(out.data.new(out.data.shape[0],self.out_chan-self.in_chan,out.data.shape[2],out.data.shape[3]).fill_(0)) 
            out =out + torch.cat([x, padding], dim=1)
        return out






ShuffleNetSettings = collections.NamedTuple("ShuffleNetSettings", "num_units, init_g1, init_g2, bottle_factor")
class ShuffleNet(SerializableModule):
    @classmethod
    def from_args(cls,ctx, args):
        settings=ShuffleNetSettings(num_units=args.shufflenet_num_units, init_g1= args.shufflenet_init_g1, init_g2=args.shufflenet_init_g2,bottle_factor=args.shufflenet_bottle_factor)
        return cls(ctx, settings)

    def __init__(self, ctx, settings):
        super().__init__()
        layer_dict=collections.OrderedDict()
        for i in range(settings.num_units):
               bc = settings.init_g1*settings.init_g2
               ic = settings.bottle_factor*bc
               oc = settings.bottle_factor*bc
               unit_settings = ShuffleNetUnitSettings(groups1 = settings.init_g1, groups2 = settings.init_g2, in_chan = ic, out_chan = oc, bottle_chan = bc)
               to_add=ShuffleNetUnit(ctx, unit_settings) 
               layer_dict["shuffle_unit{}".format(i)]=to_add


        self.sequential = nn.sequential(layer_dict)

    def forward(self, x):
        x = self.sequential(x)
        x = torch.mean(x, dim=3)
        x = torch.mean(x, dim=2)
        return x


def add_shufflenet_args(parser):
    parser.add_argument("shufflenet_num_units", type= int)
    parser.add_argument("shufflenet_init_g1", type=int)
    parser.add_argument("shufflenet_init_g2", type=int)
    parser.add_argument("shufflenet_bottle_factor", type=int)

