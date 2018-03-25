#from https://github.com/daemon/vdpwi-nn-pytorch/blob/master/vdpwi/utils/tb.py
import datetime
import sys
import os.path
import logging

from tensorboardX import SummaryWriter
from . import reporting
import torch

class TBWriter(object):
    def __init__(self, run_name_fmt="run_{}"):
        self.run_name = run_name_fmt.format(reporting.timestamp())
        self.writer = SummaryWriter(os.path.join("./tb_logs/",reporting.daystamp(),self.run_name))
        self.train_idx = 0
        self.acc_idx = 0
        self.dps_index=0
        self.param_dif_idx=0
        self.lr_idx=0
        self.upp_idx=0
        self.mult_idx=0

    def write_hyperparams(self):
        #self.writer.add_text("hyperparams", " ".join(sys.argv))
        self.writer.add_text("{}/hyperparams".format(self.run_name), " ".join(sys.argv))

    def write_num_trainable_params(self,num_trainable_params):
        #self.writer.add_text("trainable_params", str(num_trainable_params)  )
        self.writer.add_text("{}/trainable_params".format(self.run_name), str(num_trainable_params)  )

    def write_train_loss(self, loss):
        #self.writer.add_scalar("train_loss", loss, self.train_idx)
        self.writer.add_scalar("{}/train_loss".format(self.run_name), loss, self.train_idx)
        self.train_idx += 1

    def write_accuracy(self, acc):
        #self.writer.add_scalar("accuracy", acc , self.acc_idx)
        self.writer.add_scalar("{}/accuracy".format(self.run_name), acc , self.acc_idx)
        self.acc_idx += 1

    def write_data_per_second(self,dps):
        #self.writer.add_scalar("datapoints_per_second", dps , self.dps_index)
        self.writer.add_scalar("{}/datapoints_per_second".format(self.run_name), dps , self.dps_index)
        self.dps_index += 1

    def write_lr(self, lr):
        self.writer.add_scalar("{}/lr".format(self.run_name), lr , self.lr_idx)
        self.lr_idx+=1

    def write_param_change(self,params, oldparams):
        '''
            Args:
                params: list of tuples of paramters names and tensors with current values of paramters
                old_params: list of tuples of paramter names and tensors with old values of paramters
        '''
        for i, param in enumerate(params):
            dif=torch.norm(param[1]-oldparams[i][1]/param[1])
            if dif == 0:
                logging.warning("Parameter "+ param[0] +" did not change.")
            self.writer.add_scalar("prctchange/in_{}".format(param[0]), dif, self.param_dif_idx)
        self.param_dif_idx+=1
    
    def write_unpruned_params(self, upp):
        self.writer.add_scalar("{}/unpruned_masks".format(self.run_name), upp , self.upp_idx)
        self.upp_idx+=1 

def write_multiplies(self, mults):
        self.writer.add_scalar("{}/approx_multiplies".format(self.run_name), mult , self.mult_idx)
        self.mult_idx+=1 
