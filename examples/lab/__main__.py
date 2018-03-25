import argparse
import logging
import sys
import time
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import pickle

from . import basic_classify
from .genutil import modules
from .genutil import arguments
from .datatools import basic_classification
#general rule: all used modules should be able to created just by passing args
#todo: 
#implement an ensembling system that saved paramters and models in acompressed file together.  Better for organization
#thought: could we fit segmentation and sequence-to-sequence in the paradigmn of classification?  To do so we would need to allow more than none "class" per data item.  This would be the world of the output sentence for sequence-to-sequence and the pixel classes for segmentation

def initial_parser(parser = None):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, choices=["classification", "sequence_to_sequence"], default="classification")
    return parser  

def default_parser(parser=None):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_epochs",type=int,default=4)
    parser.add_argument("--validation_set_size",type=int,default=1000)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--holdout_size", type=int, default=500)
    parser.add_argument("--model_save_path",type=str, default= "./saved_models/") 
    parser.add_argument("--resume_mode", type=str, choices=["none", "standard", "ensemble"], default= "none" )
    parser.add_argument("--res_file",type=str, default="recent_model") 
    parser.add_argument("--mode", type=str, choices=["test", "train"], default="train")

    parser.add_argument("--test_report_filename", type=str)
    parser.add_argument("--use_saved_processed_data", action="store_true")
    parser.add_argument("--processed_data_path",type=str,default="../saved_processed_data")
    parser.add_argument("--report_path",type=str,default="../reports")
    parser.add_argument("--batch_size", type= int, default=32)
    parser.add_argument("--param_report", action="store_true")
    parser.add_argument("--mod_report",action="store_true")
    parser.add_argument("--proxy_report",action="store_true")
    parser.add_argument("--param_difs", action="store_true" )
    parser.add_argument("--optimizer", type=str, choices=["sgd", "rmsprop", "adam"], default="sgd")
    parser.add_argument("--init_lr",type=float, default=0.1)
    parser.add_argument("--sgd_momentum",type=float, default=0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0)
    parser.add_argument("--plateau_lr_scheduler_patience",type=int, default=10)
    parser.add_argument("--lr_scheduler",type=str, choices=[None, "exponential", "plateau", "linear", "multistep", "epoch_anneal"], default="exponential")
    parser.add_argument("--lr_gamma",type=float, default=0.99)
    parser.add_argument("--linear_scheduler_max_epoch", type=int, default=300)
    parser.add_argument("--linear_scheduler_subtract_factor", type=float, default=0.99)
    parser.add_argument("--multistep_scheduler_milestone1", type=int, default=150)
    parser.add_argument("--multistep_scheduler_milestone2", type=int, default=225)
    parser.add_argument("--epoch_anneal_numcycles", type=int, default=6)
    parser.add_argument("--epoch_anneal_mult_factor",type=int, default=1)
    parser.add_argument("--epoch_anneal_init_period",type=int, default=-1) #setting this will override numcycles
    parser.add_argument("--epoch_anneal_update_previous_incarnation",action="store_true") 
    parser.add_argument("--epoch_anneal_start_ba_after_epoch",action="store_true") 
    parser.add_argument("--epoch_anneal_reinit_after_cycle",action="store_true") 




    parser.add_argument("--grad_norm_clip",type=float, default=None)
    parser.add_argument("--output_level", type=str, choices=["info", "debug"], default="info") 
    parser.add_argument("--ensemble_args_files", type=str, nargs="+")
    
    parser.add_argument("--ensemble_autogen_args", action="store_true")# for the autogen case  
    parser.add_argument("--ensemble_models_files", type=str, nargs="+")
    parser.add_argument("--epoch_anneal_save_last", action="store_true")
    parser.add_argument("--weight_ensemble_on_validation_set", action="store_true")

    parser.add_argument("--born_again_enable", action="store_true")
    parser.add_argument("--born_again_model_file", type=str)
    parser.add_argument("--born_again_args_file",type=str, default=None)

    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--data_par_enable", action="store_true") #currently disabled
    parser.add_argument("--data_par_devices", type=int, nargs="+")
    
    parser.add_argument("--enable_pruning",action="store_true") 
    parser.add_argument("--prune_target",type=int, default=50000)
    parser.add_argument("--prune_epoch_freq", type=int, default=1)
    parser.add_argument("--prune_warmup_epochs", type=int, default=10)

    parser.add_argument("--enable_l0reg",action = "store_true")
    parser.add_argument("--l0reg_lambda", type=float, default =1.5 / 50000 )


    parser.add_argument("--report_unpruned",action="store_true") 
    
    parser.add_argument("--proxy_context_type", type=str, choices=["no_context","identity_context", "prune_context", "group_prune_context", "l0reg_context", "tanhbinarize_context", "stdfactorize_context" ], default="no_context")

    parser.add_argument("--use_nograd",action="store_true") #use nograd instead of volatile 

    #prune trained model
    parser.add_argument("--prune_trained", action="store_true", help= "Prune a trained model, then resave it ")
    parser.add_argument("--prune_trained_pct", type=int, help="pct of weights to prune")

    parser.add_argument("--validate_fr", action="store_true",help="get accuracy of a result report using ground truth")
    parser.add_argument("--validate_fr_reportfile",help="report file to validate")
    parser.add_argument("--validate_fr_truthfile", help="Ground truth file for validations")
    parser.add_argument("--validate_fr_truthfiletype",choices=["pickle"], default="pickle" )

    parser.add_argument("--clamp_all_params",action="store_true")
    parser.add_argument("--clamp_all_min",type=int)
    parser.add_argument("--clamp_all_max",type=int)



    parser.add_argument("--use_no_grad",action="store_true")


    parser.add_argument("--count_multiplies", action="store_true")
    parser.add_argument("--count_multiplies_every_cycle",action="store_true")


    parser.add_argument("--use_all_params",action="store_true") #ie optimize over mask params
    
    parser.add_argument("--factorize_trained", action="store_true", help="factorize a trained model, then resave it")
    parser.add_argument("--factorize_trained_method", choices=["svd"], default="svd")
    parser.add_argument("--factorize_svd_rank_prop",type=float, default=0.5)

    return parser


def get_args_from_files(filenames):
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args_list=[]
   for filename in filenames:
        args = arguments.parse_from_file(filename, parser)
        args_list.append(args)
   return args_list





def main():
   



   logging.basicConfig(level=logging.INFO)
   logging.info("command line options:"  )
   logging.info(" ".join(sys.argv))
   iparser = initial_parser()
   [initial_args, remaining_vargs ] = iparser.parse_known_args()
   if initial_args.paradigm == "classification":
    parser=default_parser()
    parser=basic_classify.add_args(parser)
    args = parser.parse_args(remaining_vargs)
    if args.validate_fr: #dont build a model.  Just evaluate a report
        if args.validate_fr_truthfiletype =="pickle":
         with open(args.validate_fr_truthfile, 'rb') as f: 
              dummy =pickle.load(f)
              truth= pickle.load(f)



         acc = basic_classification.score_report(args.validate_fr_reportfile, truth)
        print("ACCURACY ON GROUND TRUTH: ",acc ) 
        return

    if args.resume_mode == "ensemble":
      if args.ensemble_autogen_args:
            args_list=[]
            for filename in args.ensemble_models_files:
                cur_args=copy.deepcopy(args)
                cur_args.res_file=filename
                args_list.append(cur_args)
      else:
        args_list=get_args_from_files(args.ensemble_args_files)
        if args.ensemble_models_files is not None:
             for i,filename in enumerate(args.ensemble_models_files):
                args_list[i].res_file=filename
      if args.weight_ensemble_on_validation_set:
          for arg_instance in args_list:
              arg_instance.weight_ensemble_on_validation_set=True
      basic_classify.run(args_list, ensemble_test=True)
      return


   if args.param_report:
       show_params()
       return
   if args.mod_report:
       show_submods()
       return
   if args.proxy_report:
       show_proxies()
       return
   if args.output_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
   basic_classify.run(args)





def show_params(input_size=(32,3,32,32)):
   logging.basicConfig(level=logging.DEBUG)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   if args.proxy_context_type == "no_context": 
       for name, param in context.model.named_parameters():
         print(name)
         print(param.shape)
         print(param.requires_grad)
         if param.is_cuda:
          print("device:")
          print(param.get_device())
       param_count = modules.count_trainable_params(context.model)
   else:
        param_count = modules.count_elem(context.model.proxy_ctx.list_model_params() ) 
   print("total trainable params:{}".format(param_count)) 

def show_proxies():
   logging.basicConfig(level=logging.DEBUG)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   for proxy in context.model.proxy_ctx.list_proxies():
       print(proxy)
  
    

def show_submods():
   logging.basicConfig(level=logging.DEBUG)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   print("layers:")
   print(list(context.model.children()))


if __name__ == "__main__":
    main()
    #show_params()
