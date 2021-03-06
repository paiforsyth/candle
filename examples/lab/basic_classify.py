import argparse
import math
import logging
import time
import collections
import os.path
import pickle
import copy
import torch.utils.data as data
import torch.nn as nn
import torch.nn.utils.clip_grad
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler
import numpy as np
import functools
from contextlib import suppress
from torch.autograd import Variable



from  .datatools import set_simp 
from .datatools import set_polarity
from  .datatools import set_cifar_challenge
from  .datatools import sequence_classification
from  .datatools import basic_classification
from .datatools import img_tools
from .datatools.basic_classification import DataType
from  .datatools import word_vectors
from  .modules import maxpool_lstm
from  .modules import squeezenet
from  .modules import kim_cnn
from  .modules import coupled_ensemble
from .modules import countmult
from .monitoring import reporting
from .monitoring import tb_log
from  .genutil import modules as genutil_modules
from  .genutil import optimutil
from .modules import reset_masks
from . import __main__ as mainfuncs
from  .modules import saveable_data_par 
from .genutil import modules
from torchvision import transforms
import torchvision.datasets as tvds
import candle.prune
import candle.proxy
def add_args(parser):
    if parser is None:
        parser= argparse.ArgumentParser() 
    parser.add_argument("--dataset_for_classification",type=str,choices=["simple","moviepol", "mnist", "cifar_challenge", "cifar10", "minicifar10"],default="simple")

    parser.add_argument("--ds_path", type=str,default=None)
    parser.add_argument("--fasttext_path", type=str,default="../data/fastText_word_vectors/" )
    parser.add_argument("--data_trim", type=int, default=30000)
    parser.add_argument("--lstm_hidden_dim", type = int, default =300)
    parser.add_argument("--maxlstm_dropout_rate", type = int, default = 0.5)
    parser.add_argument("--reports_per_epoch", type=int,default=10)
    parser.add_argument("--save_prefix", type=str,default=None)
    parser.add_argument("--model_type", type=str, choices=["maxpool_lstm_fc", "kimcnn", "squeezenet", "shufflenet"],default="maxpool_lstm_fc")
    parser.add_argument("--cifar_random_erase", action="store_true")
    parser.add_argument("--classification_loss_type",type=str, choices=["cross_entropy", "nll", "square_hinge"], default="cross_entropy")
    parser.add_argument("--coupled_ensemble",type=str, choices=["on", "off"], default="off")
    parser.add_argument("--coupled_ensemble_size", type=int, default=4)
    
    

    parser.add_argument("--cifar_shuffle_val_set",action="store_true")
    
    parser.add_argument("--use_custom_test_data_file",action="store_true")
    parser.add_argument("--custom_test_data_file")
    parser.add_argument("--num_custom_test_file_points", type=int, default=1000)

    parser.add_argument("--multi_score_model",action="store_true") #for use with models that, like branchynet, prduce multiple score outputs at train time
    parser.add_argument("--multi_score_unit_weighting",action="store_true")#losses all get same weight
    parser.add_argument("--multi_score_loss_weighting",nargs="+" ) #weight for the losses derived from each of the scores 

    parser.add_argument("--use_val_as_test",action="store_true")

    kim_cnn.add_args(parser)
    squeezenet.add_args(parser)
    
    return parser


class Context:
    def __init__(self, model, train_loader, val_loader, optimizer,indexer, category_names, tb_writer, train_size, data_type, scheduler, test_loader,cuda, holdout_loader, num_categories, model_parameters):
        self.model=model
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.holdout_loader= holdout_loader
        self.optimizer=optimizer
        self.categpry_names=category_names
        self.tb_writer=tb_writer
        self.train_size=train_size
        self.data_type=data_type
        self.scheduler=scheduler
        self.test_loader=test_loader
        self.cuda=cuda
        self.num_categories=num_categories
        self.model_parameters = model_parameters

        
        self.stashfile=None

    def stash_model(self):
        self.stashfile = "../temp/temp_storage_"+str(id(self))
        torch.save(self.model, self.stashfile)
        self.model=None
    def unstash_model(self):
        self.model=torch.load(self.stashfile)
        if self.cuda:
            self.model=self.model.cuda()
        self.stashfile=None


def make_context(args):
   holdout_loader =None
   if args.enable_l0reg or args.proxy_context_type == "l0reg_context" :
       assert  args.enable_l0reg and args.proxy_context_type == "l0reg_context" 
   if args.enable_l1reg or args.proxy_context_type == "l1reg_context_slimming":
       assert args.enable_l1reg and args.proxy_context_type == "l1reg_context_slimming"
   assert args.save_prefix is not None 
   if args.dataset_for_classification == "simple":
        if args.save_prefix is None:
            args.save_prefix="simplification_classification"
        if args.ds_path is None:
            args.ds_path= "../data/sentence-aligned.v2" 
        train_dataset, val_dataset, index2vec, indexer = set_simp.load(args)
        category_names={0:"normal",1:"simple"}
        data_type=DataType.SEQUENCE
   elif args.dataset_for_classification == "moviepol":
        if args.save_prefix is  None:
            args.save_prefix= "moviepol"
        if args.ds_path is None:
            args.dspath = "../data/rt-polaritydata"
        train_dataset, val_dataset, index2vec, indexer = set_polarity.load(args)
        category_names={0:"negative",1:"positive"}
        data_type=DataType.SEQUENCE
   elif args.dataset_for_classification == "mnist":
        num_categories =10
        train_dataset = tvds.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))]))
        val_dataset = tvds.MNIST('../data/mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))]))
        category_names={0:"1",1:"2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
        data_type = DataType.IMAGE 
        test_dataset=val_dataset #for testing
   elif args.dataset_for_classification == "cifar_challenge":
        num_categories = 100
        data_type = DataType.IMAGE
        f=open("./local_data/cifar/train_data","rb")
        squashed_images=pickle.load(f)
        labels=pickle.load(f)
        f.close()
        train_dataset,val_dataset = set_cifar_challenge.make_train_val_datasets(squashed_images, labels, args.validation_set_size, transform=None, shuf=args.cifar_shuffle_val_set) 
        tr = transforms.Compose([transforms.RandomCrop(size=32 ,padding= 4), transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
        if args.cifar_random_erase:
                tr=transforms.Compose([tr, img_tools.RandomErase()])
        if args.holdout:
                holdout_dataset, val_dataset = val_dataset.split(args.holdout_size)
        train_dataset.transform = tr
        val_dataset.transform = transforms.ToTensor()
       
        if args.mode == "train":
            pass
            test_dataset=None
        elif args.mode == "test":
            if args.use_custom_test_data_file:
                f=open(args.custom_test_data_file,"rb")
            else:
                f=open("./local_data/cifar/test_data","rb")
            squashed_images=pickle.load(f)[:args.num_custom_test_file_points]
            test_dataset= set_cifar_challenge.Dataset(data=squashed_images, labels=[-1]*squashed_images.shape[0], transform=transforms.ToTensor())
            f.close()


        category_names= { k:v for k,v in enumerate(set_cifar_challenge.CIFAR100_LABELS_LIST)}
   elif args.dataset_for_classification == "cifar10":
        num_categories = 10
        data_type = DataType.IMAGE
        tr = transforms.Compose([transforms.RandomCrop(size=32 ,padding= 4), transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
        if args.cifar_random_erase:
            tr=transforms.Compose([tr, img_tools.RandomErase()])

        f=open('./local_data/cifar10/cifar-10-batches-py/data_batch_1','rb')
        dictionary=pickle.load(f,encoding="bytes")
        squashed_images = dictionary[b'data']
        labels = dictionary[b'labels']
        f.close()
        for i in range(2,6):
                f=open('local_data/cifar10/cifar-10-batches-py/data_batch_'+str(i),'rb')
                dictionary = pickle.load(f, encoding='bytes')
                squashed_images = np.concatenate((squashed_images, dictionary[b'data']),axis=0)
                labels.extend(dictionary[b'labels'])
                f.close()

        train_dataset, val_dataset = set_cifar_challenge.make_train_val_datasets(squashed_images, labels, args.validation_set_size, transform=None, shuf=args.cifar_shuffle_val_set) 
        train_dataset.transform = tr
        val_dataset.transform = transforms.ToTensor()
        f=open('./local_data/cifar10/cifar-10-batches-py/test_batch','rb')
        dictionary=pickle.load(f,encoding="bytes")
        squashed_images = dictionary[b'data']
        labels = dictionary[b'labels']
        f.close()
        test_dataset= set_cifar_challenge.Dataset(data=squashed_images, labels=labels, transform=transforms.ToTensor())
        if args.use_val_as_test:
            test_dataset=val_dataset

        
        #train_dataset = tvds.CIFAR10("./local_data/cifar10/", train=True, download= True, transform=tr ) 
        #val_dataset = tvds.CIFAR10("./local_data/cifar10/", train=False, download= True, transform=transforms.ToTensor() ) 
        category_names = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8: "ship", 9: "truck" }
        if args.use_custom_test_data_file:
                f=open(args.custom_test_data_file,"rb")
                dictionary = pickle.load(f,encoding='bytes')
                squashed_images=dictionary[b'data'][:args.num_custom_test_file_points]
                test_dataset= set_cifar_challenge.Dataset(data=squashed_images, labels=dictionary[b'labels'][:args.num_custom_test_file_points], transform=transforms.ToTensor())
                f.close()
   elif args.dataset_for_classification == "minicifar10":
        minicifar_size=200+args.validation_set_size 
        num_categories = 10
        data_type = DataType.IMAGE
        tr = transforms.Compose([transforms.RandomCrop(size=32 ,padding= 4), transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
        if args.cifar_random_erase:
            tr=transforms.Compose([tr, img_tools.RandomErase()])

        f=open('./local_data/cifar10/cifar-10-batches-py/data_batch_1','rb')
        dictionary=pickle.load(f,encoding="bytes")
        squashed_images = dictionary[b'data']
        labels = dictionary[b'labels']
        f.close()
        for i in range(2,6):
                f=open('local_data/cifar10/cifar-10-batches-py/data_batch_'+str(i),'rb')
                dictionary = pickle.load(f, encoding='bytes')
                squashed_images = np.concatenate((squashed_images, dictionary[b'data']),axis=0)
                labels.extend(dictionary[b'labels'])
                f.close()
        #remove most images to make minicifar
        squashed_images=squashed_images[:minicifar_size,:]
        labels=labels[:minicifar_size]
        logging.info("Counts of labels in minicifar:{}".format(collections.Counter(labels[args.validation_set_size:minicifar_size])))
        train_dataset, val_dataset = set_cifar_challenge.make_train_val_datasets(squashed_images, labels, args.validation_set_size, transform=None, shuf=args.cifar_shuffle_val_set) 
        train_dataset.transform = tr
        val_dataset.transform = transforms.ToTensor()

        f=open('./local_data/cifar10/cifar-10-batches-py/test_batch','rb')
        dictionary=pickle.load(f,encoding="bytes")
        squashed_images = dictionary[b'data']
        labels = dictionary[b'labels']
        f.close()
        test_dataset= set_cifar_challenge.Dataset(data=squashed_images, labels=labels, transform=transforms.ToTensor())
        if args.use_val_as_test:
            test_dataset=val_dataset

        
        #train_dataset = tvds.CIFAR10("./local_data/cifar10/", train=True, download= True, transform=tr ) 
        #val_dataset = tvds.CIFAR10("./local_data/cifar10/", train=False, download= True, transform=transforms.ToTensor() ) 
        category_names = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8: "ship", 9: "truck" }





   else:
        raise Exception("Unknown dataset.")
   


   logging.info("using save prefix "+str(args.save_prefix))


   if data_type == DataType.SEQUENCE:
        embedding=word_vectors.embedding(index2vec, indexer.n_words,300)
        train_loader= data.DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True,collate_fn = sequence_classification.make_collater(args))
        val_loader= data.DataLoader(val_dataset,batch_size = args.batch_size, shuffle = False, collate_fn = sequence_classification.make_collater(args))
   elif data_type == DataType.IMAGE:
       indexer= None
       if args.mode == "train": #can probably remove this if block.  Now we have all loaders availible in all modes
            train_loader=data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=basic_classification.make_var_wrap_collater(args), drop_last= args.drop_last_training_batch)
            val_loader=data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args,volatile=True ))
            test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args, volatile=True)) if test_dataset is not None else None
       elif  args.mode == "test":
            train_loader=data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=basic_classification.make_var_wrap_collater(args))
            test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args, volatile=True))
            val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args)) #certain ensemble methods use the val dataset 
            assert(args.resume_mode == "standard" or args.resume_mode == "ensemble")
            # if args.holdout:
                    # holdout_loader=data.DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args,volatile=True))
   else:
       raise Exception("Unknown data type.")
        
   
   if args.model_type == "maxpool_lstm_fc":
    model=maxpool_lstm.MaxPoolLSTMFC.from_args(embedding, args) 
   elif args.model_type == "kimcnn":
       model=kim_cnn.KimCNN.from_args(embedding,args) 
   elif args.model_type == "squeezenet":
       model=squeezenet.SqueezeNet.from_args(args)
       if args.l0reg_lambda_vary_by_layer:
           model.set_lambd_by_subblock(lambd_first=args.l0reg_lambda_vary_first, lambd_last=args.l0reg_lambda_vary_last)
       if args.l0reg_lambda_vary_by_sublayer_name:
            name_dict={}
            name_dict["squeeze_conv"]=args.l0reg_lambda_squeeze_conv
            name_dict["group_conv"]= args.l0reg_lambda_group_conv
            name_dict["expand_conv"]=args.l0reg_lambda_expand_conv
            model.set_l0_lambda_by_name(name_dict)

   else:
       raise Exception("Unknown model")


   if args.cuda and not args.data_par_enable:
       model=model.cuda()
   if args.coupled_ensemble =="on":
        assert args.classification_loss_type == "nll"
        model_list = []
        for i in range(args.coupled_ensemble_size):
            cur_model=copy.deepcopy(model)
            cur_model.init_params()
            model_list.append(cur_model)
        model = coupled_ensemble.CoupledEnsemble(model_list)
   elif args.coupled_ensemble != "off":
        raise Exception("Unknown coupled ensemble settting")

   if args.proxy_context_type == "no_context":
       model_parameters = model.parameters()
   else:
       if args.enable_l0reg:
           assert args.use_all_params
       elif args.enable_l1reg:
           assert not args.use_all_params #in the network slimming case, we do not optimize over the masks.  They are set to zero based on a pruning scheule
       if args.use_all_params:
           model_parameters = model.proxy_ctx.list_params()
       else: 
            model_parameters = model.proxy_ctx.list_model_params()
        

   if args.optimizer == "sgd":
        optimizer=optim.SGD(model_parameters,lr=args.init_lr, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay )
       
   elif args.optimizer == "rmsprop":
       optimizer = optim.RMSprop(model_parameters, lr=args.init_lr)
   elif args.optimizer == "adam":
       optimizer = optim.Adam(model_parameters, lr=args.init_lr)
   else:
       raise Exception("Unknown optimizer.") 

   if args.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)
   elif args.lr_scheduler == "plateau":
       scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", verbose=True, patience=args.plateau_lr_scheduler_patience)
   elif args.lr_scheduler == "linear":
        lam = lambda epoch: 1-args.linear_scheduler_subtract_factor* min(epoch,args.linear_scheduler_max_epoch)/args.linear_scheduler_max_epoch 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam )
   elif args.lr_scheduler == "multistep":
        milestones=[args.multistep_scheduler_milestone1, args.multistep_scheduler_milestone2]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma )
   elif args.lr_scheduler == "epoch_anneal":
       if args.epoch_anneal_init_period>0:
           Tmax = args.epoch_anneal_init_period
       else:
           Tmax=args.num_epochs//args.epoch_anneal_numcycles
       scheduler= optimutil.MyAnneal(optimizer=optimizer, Tmax=Tmax,  init_lr=args.init_lr)
   elif args.lr_scheduler is None or args.lr_scheduler == "none":
       scheduler = None
   else: 
       raise Exception("Unknown Scheduler")


   if args.mode =="train":
       pass
       train_size=len(train_dataset)
       # test_loader = None
   elif args.mode=="test":
       pass
       train_size= None
       # train_loader = None

   return Context(model, train_loader, val_loader, optimizer, indexer, category_names=category_names, tb_writer=tb_log.TBWriter("{}_"+args.save_prefix), train_size=train_size, data_type=data_type, scheduler=scheduler, test_loader=test_loader, cuda=args.cuda, holdout_loader= holdout_loader, num_categories = num_categories, model_parameters=model_parameters)






def run(args, ensemble_test=False):
   if ensemble_test:
       assert type(args) is list
       contexts=[] #[make_context(arg_instance) for arg_instance in args ]
       for arg_instance in args:
           contexts.append(make_context(arg_instance))
           contexts[-1].stash_model()
       for context, arg_instance in zip(contexts,args):
            logging.info("loading saved model from file: "+arg_instance.res_file)
            context.unstash_model()
            context.model.load(os.path.join(arg_instance.model_save_path, arg_instance.res_file))
            context.stash_model()
       if args[0].weight_ensemble_on_validation_set:
           meta_model = basic_classification.optimize_ensemble_on_val(contexts, contexts[0].val_loader)
       else:
            meta_model = None
       basic_classification.make_ensemble_prediction_report(contexts, contexts[0].test_loader, args[0].test_report_filename, meta_model=meta_model)
       return

   context=make_context(args) 
   if args.resume_mode == "standard":
       logging.info("loading saved model from file: "+args.res_file)
       context.model.load(os.path.join(args.model_save_path, args.res_file), strict= not args.load_nonstrict)
   if args.adjust_out_dim_after_loading:
        context.model.adjust_out_dim(args.new_out_dim, linear=args.new_final_linear,spatial_dim=args.squeezenet_final_side_length )
   if args.born_again_enable:
       if args.born_again_args_file is not None:
            logging.info("loading born again args from "+args.born_again_args_file)
            previous_incarnation_context=make_context( mainfuncs.get_args_from_files([args.born_again_args_file]) [0])
       else:
            previous_incarnation_context=make_context(args)
       logging.info("loading previous incarnation from file: "+str(args.born_again_model_file))
       previous_incarnation_context.model.load(os.path.join(args.born_again_model_file))
       for param in previous_incarnation_context.model.parameters():
            param.requires_grad = False
   if args.print_model:
        logging.info(repr(context.model))
        return

   if args.reset_masks_after_loading:
        reset_masks.reset_masks(context.model)
        

   if args.plot_unpruned_masks:
        import matplotlib.pyplot as plt
        import numpy as np
        plot_dict = context.model.prop_nonzero_masks()    
        fig, ax = plt.subplots()
        for label, props in plot_dict.items():
            layer_nums = np.arange(0,len(props))
            np_props = np.asarray(props)
            ax.plot(layer_nums, np_props, label=label)
        legend = ax.legend(loc='upper left')
        ax.set(xlabel="residual unit #", ylabel = "proportion of masks unpruned", title=args.plot_title)
        plt.show()
        return

   if args.plot_flop_reduction_by_layer:
        import matplotlib.pyplot as plt
        import numpy as np
        img_h, img_w, channels = get_dims_from_dataset(args.dataset_for_classification)   
        props, _ =  context.model.prop_flop_reduction(img_h = img_h, img_w= img_w, input_channels = channels)    
        fig, ax = plt.subplots()
        layer_nums = np.arange(0,len(props))
        np_props = np.asarray(props)
        ax.plot(layer_nums, np_props)
        ax.set(xlabel="residual unit #", ylabel = "pruned flops / unpruned flops", title=args.plot_title)
        plt.show()
        return

   if args.plot_absolute_flop_reduction_by_layer:
        import matplotlib.pyplot as plt
        import numpy as np
        img_h, img_w, channels = get_dims_from_dataset(args.dataset_for_classification)   
        _, difs =  context.model.prop_flop_reduction(img_h = img_h, img_w= img_w, input_channels = channels)    
        fig, ax = plt.subplots()
        layer_nums = np.arange(0,len(difs))
        np_difs = np.asarray(difs)
        ax.plot(layer_nums, np_difs)
        ax.set(xlabel="residual unit #", ylabel =  "unpruned flops - pruned_flops", title=args.plot_title)
        plt.show()
        return




   if args.mode == "test":
        basic_classification.make_prediction_report(context, context.test_loader,args.test_report_filename, no_grad=args.use_nograd)  
        if args.dataset_for_classification == "cifar_challenge" or args.dataset_for_classification == "cifar10":
           img_h=32
           img_w=32
           channels=3
        logging.info("multiplies performed by tested model "+ str(countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)))    

        return
   
   if args.lr_scheduler == "epoch_anneal":
        epoch_anneal_cur_cycle=0
        
   
   context.tb_writer.write_hyperparams()
   timestamp=reporting.timestamp()
   
   report_interval=max(len(context.train_loader) //  args.reports_per_epoch ,1)
   accumulated_loss=0 
   if args.enable_l1reg:
       accumulated_l1l=0
   if args.enable_l2reg_stochastic:
       accumulated_l2l_stochastic=0

   param_count=genutil_modules.count_trainable_params(context.model)
   if args.proxy_context_type == "no_context": 
        param_count = modules.count_trainable_params(context.model)
   else:
        param_count = modules.count_elem(context.model.proxy_ctx.list_model_params() ) 

   logging.info("Number of parameters: "+ str(param_count))
   context.tb_writer.write_num_trainable_params(param_count)


   if args.factorize_trained:
       context.model.eval()
       if args.dataset_for_classification == "cifar_challenge" or args.dataset_for_classification == "cifar10":
           img_h=32
           img_w=32
           channels=3
       logging.info("multiplies before factorization: "+ str(countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)))    
       
       if args.factorize_trained_method == "svd":
            with torch.no_grad():
             logging.info("svd factorizing model")
             context.model.proxy_ctx.save_samples_all()
             for i,(batch_in, *other) in enumerate(context.train_loader): 
                    if i>=20:
                        break
                    context.model(batch_in)
             context.model.proxy_ctx.factorize_all(strategy="svd",rank_prop=args.factorize_svd_rank_prop) 
             context.model.proxy_ctx.clear_samples_all()
             logging.info("multiplies after factorization: "+ str(countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)))    
             context.model.save(os.path.join( args.model_save_path, args.res_file+"_svd_factorize_" + str(args.factorize_svd_rank_prop) )  )
       else:
            raise Exception("Unknown factorization method")
       return

   if args.count_multiplies:
       context.model.eval() #for sampling to compute avg mults in forking models
       if args.get_forking_props_on_val:
            squeezenet.forking_props_from_sample(context.model,context.val_loader )
       if args.count_mult_override_img_dims:
           img_h = args.count_mult_override_imgh
           img_w = args.count_mult_override_imgw
           channels=3
       elif args.dataset_for_classification == "cifar_challenge" or args.dataset_for_classification =="cifar10":
           img_h=32
           img_w=32
           channels=3
       cm = countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)
       if  args.short_test_report:
           print(cm)
       else:
           print("Approx number of multiplies: ",cm )    
       return


   if args.set_prune_deemph:
       context.model.deemph_string(args.prune_deemph_string)

   if args.enable_pruning:
        init_mask_count = context.model.proxy_ctx.count_unpruned_masks()
        logging.info("Initial number of masks {}".format(init_mask_count))
        if args.autocalc_prune_unit:
            if args.prune_calc_type =="relative":
                assert not args.sense_adaptive_pruning  
                prune_unit = math.ceil((1- (args.prune_target_frac)**(1/args.prune_phase_duration))*100)
                logging.info("relative prune unit is {}".format(prune_unit))
            elif args.prune_calc_type =="absolute":
                assert args.sense_adaptive_pruning
                prune_abs_unit = math.ceil( (1-args.prune_target_frac)/args.prune_phase_duration*100  )
                logging.info("absolute prune unit is {}".format(prune_abs_unit))
        else:
            prune_unit = args.prune_unit
            prune_abs_unit= prune_unit
        if args.prune_target_frac is not None:
            prune_target = int(init_mask_count *args.prune_target_frac )
        else:
            prune_target =args.prune_target
        logging.info("Target number of masks is : {}".format(prune_target))

   if args.prune_trained:
    if args.prune_trained_hz:
             assert args.hz_lasso_target_prop is None #use prune_trained_pct 
             before_score=basic_classification.evaluate(context, context.val_loader,no_grad=args.use_nograd)
             logging.info("accuracy before hz_lasso: {} ".format(before_score ) )
             hz_loader = context.train_loader if args.hz_lasso_use_train_loader else context.val_loader
             prop =(100 -args.prune_trained_pct)/100
             hz_lasso_whole_model(context, args,num_samples= args.hz_lasso_num_samples,target_prop= prop, loader=hz_loader,solve_for_weights =args.hz_lasso_solve_for_weights)
             after_score=basic_classification.evaluate(context, context.val_loader,no_grad=args.use_nograd)
             logging.info("accuracy after hz_lasso{}".format(after_score))
             context.model.save(os.path.join( args.model_save_path, args.res_file+"_prune_" + str(args.prune_trained_pct) )  )
             if not  args.short_test_report:
                context.model.display_subblock_nonzero_masks(nodesc=args.verbose_prune_trained_nodesc)
             return
    elif args.group_prune_strategy == "taylor":
        #logging.info("pruning trained model using taylor method")
        taylor_sample_batches(context,args)
        prunefunc = get_pruning_func(context, args)
        prunefunc(args.prune_trained_pct)
        n_unpruned = context.model.proxy_ctx.count_unpruned_masks()
        logging.info("Unpruned masks: "+str(n_unpruned))
        context.model.save(os.path.join( args.model_save_path, args.res_file+"_prune_taylor_" + str(args.prune_trained_pct) )  )
        #taylor_sample_clear(context, args)
        if args.verbose_prune_trained:
            context.model.display_subblock_nonzero_masks(nodesc=args.verbose_prune_trained_nodesc)

        return
         
    else:

       prunefunc = get_pruning_func(context, args)
       if args.iterative_prune_trained: 
           assert args.prune_absolute
           init_mask_count = context.model.proxy_ctx.count_unpruned_masks()
           target_mask_count=math.ceil(init_mask_count*(100-args.prune_trained_pct)/100 )
           while True:
               if context.model.proxy_ctx.count_unpruned_masks()<= target_mask_count:
                   break 
               prunefunc(1)
       else:
            prunefunc(args.prune_trained_pct)
      # context.model.display_subblock_nonzero_masks()
      # import pdb; pdb.set_trace()

       model_copy = copy.deepcopy(context.model)
       if args.recalc_weights_after_prune_trained:
            recalc_weights_pruned(context, args, num_samples=3, loader=context.train_loader, model_copy=model_copy)
       n_unpruned = context.model.proxy_ctx.count_unpruned_masks()
       logging.info("Unpruned masks: "+str(n_unpruned))
       context.model.save(os.path.join( args.model_save_path, args.res_file+"_prune_" + str(args.prune_trained_pct) )  )
       if args.verbose_prune_trained:
            context.model.display_subblock_nonzero_masks()
       return


   if args.sensitivity_report:
        one_layer_prune_func = get_one_layer_pruning_func(context,args,prune_abs_unit)
        accs=  by_block_accuracies(context, args, prune_abs_unit, one_layer_prune_func)
        logging.info(accs)
        return

   if args.do_condense:
        conds_so_far=0

   if args.terminate_after_pruning:
       iter_after_pruning=0

   if args.reset_weights_before_start:
       logging.info("resetting weights")
       context.model.reset_weights()


   if args.report_test_error_before_start:
        test_acc = basic_classification.evaluate(context, context.test_loader,no_grad=args.use_nograd)
        logging.info("INIITIAL TEST ACCURACY:{}".format(test_acc))
        print("INIITIAL TEST ACCURACY:{}".format(test_acc))

   if args.maintain_abs_deriv_sum:
        enable_grad_storage(context.model)

   
   best_eval_score=-float("inf")
   for epoch_count in range(args.num_epochs):
        context.model.train()
        logging.info("Starting epoch "+str(epoch_count) +".")
        if args.param_difs:
           param_tensors=genutil_modules.get_named_trainable_param_tensors(context.model)
        step=0
        epoch_start_time=time.time()
        for batch_in, *other in context.train_loader: 
            categories = other[0]
            if context.data_type == DataType.SEQUENCE:
                pad_mat = other[1]  
            step+=1

            context.optimizer.zero_grad()
            




            #for image classification, batch_in will have dimension batchsize by imagesize and scores will have dimension batchsize by number of categories
            #For sequence-to-squence batch in will have dimension batchsize by the max sequence length in the batch. scores  will have dimension batchsize by max sqeunce_length by categoreis

            scores= context.model(batch_in,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch_in)  #should have dimension batchsize by number of classes
            if args.born_again_enable:
                assert not args.multi_score_model
                context = torch.no_grad() if args.use_no_grad else suppress 
                batch_in_v=batch_in.clone()
                batch_in_v.volatile=True
                with context:
                    previous_incarnation_scores = previous_incarnation_context.model(batch_in_v,pad_mat) if previous_incarnation_context.data_type == DataType.SEQUENCE else previous_incarnation_context.model(batch_in_v)


            #move categories to same device as scores
            if not args.multi_score_model and  scores.is_cuda:
                categories=categories.cuda(scores.get_device())
            if args.multi_score_model and scores[0].is_cuda:
                categories=categories.cuda(scores[0].get_device())

            if args.classification_loss_type == "cross_entropy":
                if args.multi_score_model:
                    assert args.squeezenet_use_forking
                    if args.multi_score_unit_weighting:
                        loss=0
                        for branch_scores in scores:
                            loss+=F.cross_entropy(branch_scores,categories)
                    else:
                        raise Exception("Not implemented!")
                else:
                    loss=  F.cross_entropy(scores,categories) 
                if args.born_again_enable:
                    previous_incarnation_probs = F.softmax(previous_incarnation_scores,dim=1)
                    previous_incarnation_divergence = F.kl_div(F.log_softmax(scores,dim=1), previous_incarnation_probs )
                    loss+=previous_incarnation_divergence
            elif args.classification_loss_type == "nll":
                assert not args.born_again_enable
                assert not args.multi_score_model
                loss= F.nll_loss(scores,categories)
            elif args.classification_loss_type == "square_hinge": 
                assert not args.born_again_enable
                assert not args.multi_score_model
                mult = Variable(categories.data.new(categories.shape[0], context.num_categories).fill_(0).float()) 
                for i in range(categories.shape[0]):
                    mult[i,categories[i]]=1
                mult = 2 * mult - 1
                
                loss = torch.mean(torch.max( Variable(categories.data.new(1).fill_(0).float()), 1 - mult * scores ) ** 2)

            if args.enable_l0reg:
                lamb_param = None if (args.l0reg_lambda_vary_by_layer or args.l0reg_lambda_vary_by_sublayer_name) else args.l0reg_lambda
                loss += context.model.proxy_ctx.l0_loss(lamb_param) 

            if args.enable_l1reg and ( (not args.disable_l1_reg_after_epoch) or  epoch_count<= args.l1_reg_final_epoch ) :
                l1l = context.model.proxy_ctx.l1_loss_slimming(args.l1reg_lambda)
                accumulated_l1l+=l1l
                loss += l1l

            if args.enable_l2reg_stochastic:
                l2l_stochastic =context.model.proxy_ctx.l2_loss_stochastic(args.l2reg_stochastic_lambda) 
                accumulated_l2l_stochastic+=float(l2l_stochastic)
                loss+=l2l_stochastic
            loss.backward()
#comment
            if args.maintain_abs_deriv_sum:
                update_abs_deriv_sum(context.model)



            if args.grad_norm_clip is not None:
                torch.nn.utils.clip_grad.clip_grad_norm(context.model.parameters(), args.grad_norm_clip)
            context.optimizer.step()
            
            if args.clamp_all_params:
                for param in context.model_parameters:
                    param.data.clamp_(args.clamp_all_min,args.clamp_all_max)


            accumulated_loss+=float(loss)
            context.tb_writer.write_train_loss( float(loss)  )
            if step % report_interval == 0:
                reporting.report(epoch_start_time,step,len(context.train_loader), accumulated_loss / report_interval)
                accumulated_loss = 0
                if args.enable_l1reg:
                    logging.info("l1_loss:{}".format(accumulated_l1l/report_interval))
                    accumulated_l1l = 0
                if args.enable_l2reg_stochastic:
                    logging.info("avg l2 stochastic loss:{}".format(accumulated_l2l_stochastic/report_interval) )
                    accumulated_l2l_stochastic = 0
        #added tor try to clear computation graph after every eppoch
        del loss
        del scores
        context.model.eval()
        epoch_duration = time.time() - epoch_start_time
        context.tb_writer.write_data_per_second( context.train_size/epoch_duration)
        if args.param_difs:
            new_param_tensors=genutil_modules.get_named_trainable_param_tensors(context.model)
            context.tb_writer.write_param_change(new_param_tensors, param_tensors)
            param_tensors=new_param_tensors
        if args.maintain_abs_deriv_sum:
            disable_grad_storage(context.model)
        if epoch_count == 0 or epoch_count % args.eval_interval==0:
            logging.info("recalculating eval score")
            eval_score=basic_classification.evaluate(context, context.val_loader,no_grad=args.use_nograd)
        if args.maintain_abs_deriv_sum:
            enable_grad_storage(context.model)

        context.tb_writer.write_accuracy(eval_score)
        logging.info("Finished epoch number "+ str(epoch_count+1) +  " of " +str(args.num_epochs)+".  Accuracy is "+ str(eval_score) +".")
        if args.report_unpruned:
            n_unpruned = float(context.model.proxy_ctx.count_unpruned_masks())
            logging.info("Unpruned masks: "+str(n_unpruned))
            context.tb_writer.write_unpruned_params(n_unpruned)
        if args.show_network_strucutre_every_epoch:
                 logging.info("current model:")
                 logging.info(repr(context.model))
        if args.show_nonzero_masks_every_epoch:
            context.model.display_subblock_nonzero_masks(warning=False)
       
        if args.save_every_epoch:
            context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_most_recent" )  )
            logging.info("saving most recent model")
        

        if eval_score > best_eval_score:
            best_eval_score=eval_score
            logging.info("Saving model")
            context.model.save(os.path.join(args.model_save_path,timestamp+"recent_model" )  )
            
            if args.lr_scheduler == "epoch_anneal":
                logging.info("saving as checkpoint" + str(epoch_anneal_cur_cycle))
                context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_checkpoint_" +str(epoch_anneal_cur_cycle) )  )
            else:
                context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_best_model" )  )


        if context.scheduler is not None:
            if args.lr_scheduler == "exponential" or args.lr_scheduler == "linear" or args.lr_scheduler == "multistep":
                context.tb_writer.write_lr(context.scheduler.get_lr()[0] )
                context.scheduler.step()
            elif args.lr_scheduler == "plateau":
               # context.tb_writer.write_lr(next(context.optimizer.param_groups)['lr'] )
                context.scheduler.step(eval_score)
            elif args.lr_scheduler == "epoch_anneal":
                context.tb_writer.write_lr(context.scheduler.cur_lr() )
                context.scheduler.step()
                if context.scheduler.cur_step == context.scheduler.Tmax:
                    logging.info("Hit  min learning rate.  Restarting learning rate annealing.")
                    context.scheduler.cur_step = -1
                    context.scheduler.step()
                    best_eval_score= -float("inf")
                    if args.epoch_anneal_save_last:
                        context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_endofcycle_checkpoint_" +str(epoch_anneal_cur_cycle) )  )
                    if args.epoch_anneal_mult_factor != 1:
                        logging.info("Multiplying anneal duration by "+str(args.epoch_anneal_mult_factor))
                        context.scheduler.Tmax*=args.epoch_anneal_mult_factor 
                        logging.info("anneal duration currently:"+str(context.scheduler.Tmax))
                    if args.epoch_anneal_update_previous_incarnation:
                        if args.epoch_anneal_start_ba_after_epoch and epoch_anneal_cur_cycle == 0:
                            args.born_again_enable=True
                            previous_incarnation_context=make_context(args)

                        assert(args.epoch_anneal_save_last)
                        logging.info("loading previous incarnation")
                        previous_incarnation_context.model.load(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_endofcycle_checkpoint_" +str(epoch_anneal_cur_cycle) ) )
                        for param in previous_incarnation_context.model.parameters():
                            param.requires_grad = False
                        if args.epoch_anneal_reinit_after_cycle:
                            logging.info("resenting parameters of current model")
                            context.model.init_params()
   
                    epoch_anneal_cur_cycle+=1
                    
            else:
                raise Exception("Unknown Scheduler")

        if args.count_multiplies_every_cycle:
                if args.count_mult_override_img_dims:
                    img_h = args.count_mult_override_imgh
                    img_w = args.count_mult_override_imgw
                    channels=3
                else:
                    img_h, img_w, channels=get_dims_from_dataset(args.dataset_for_classification)
                mults = countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)
                logging.info("Approx number of multiplies: "+str(mults) )    
                context.tb_writer.write_multiplies(mults)

        if args.weight_reset_enable and epoch_count == args.weight_reset_epoch_num :
                context.model.reset_weights()
           
        if args.print_params_after_epoch:
                logging.info("model_params:")
                for param in context.model.proxy_ctx.list_model_params():
                   logging.info(str(param)) 
                logging.info("mask_params")
                for param in context.model.proxy_ctx.list_mask_params():
                    logging.info(str(param))
        if args.enable_pruning: 
             assert(args.report_unpruned)
             if args.group_prune_strategy == "taylor":
                 assert args.maintain_abs_deriv_sum 
             if epoch_count >= args.prune_warmup_epochs and epoch_count % args.prune_epoch_freq==0 and n_unpruned> prune_target:
                logging.info("pruning...")
                prunefunc = get_pruning_func(context, args)
                if args.prune_calc_type =="relative":
                    pu = prune_unit
                elif args.prune_calc_type =="absolute":
                    pu=prune_abs_unit

                prunefunc(pu)
             if n_unpruned <=prune_target and args.terminate_after_pruning:
                iter_after_pruning+=1
                if iter_after_pruning > args.iterations_after_pruning:
                    break


        if args.group_prune_strategy ==  "taylor" and args.maintain_abs_deriv_sum :
                    clear_abs_deriv_sum(context.model)
                    clear_record_of_output(context.model) #neccesary because evaluating the model may have caused additional output to be stored


        if args.do_condense and epoch_count >= args.condense_warmup and (epoch_count-args.condense_warmup) % args.condense_interval == 0 and conds_so_far<args.squeezenet_condense_num_c_groups-1:
            context.model.condense()

        if args.hz_lasso_enable and epoch_count == args.hz_lasso_at_epoch:
             logging.info("accuracy before hz_lasso: {} ".format(eval_score ) )
             context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_before_hz_lasso"  )  )
             hz_loader = context.train_loader if args.hz_lasso_use_train_loader else context.val_loader
             hz_lasso_whole_model(context, args,num_samples= args.hz_lasso_num_samples,target_prop= args.hz_lasso_target_prop, loader=hz_loader,solve_for_weights =args.hz_lasso_solve_for_weights)
             after_score=basic_classification.evaluate(context, context.val_loader,no_grad=args.use_nograd)
             logging.info("accuracy after hz_lasso{}".format(after_score))
             context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_after_hz_lasso"  )  )




   if args.show_arch_on_completion:
        logging.info(repr(context.model.children()))
   
   if args.report_test_error_at_end:
        if args.maintain_abs_deriv_sum:
            disable_grad_storage(context.model)

        test_acc = basic_classification.evaluate(context, context.test_loader,no_grad=args.use_nograd)
        logging.info("FINAL TEST ACCURACY:{}".format(test_acc))
        if args.short_run_report:
                img_h, img_w, channels=get_dims_from_dataset(args.dataset_for_classification)
                mults = countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)
                print("{},{}".format(mults,test_acc))
        else:
            print("FINAL TEST ACCURACY:{}".format(test_acc))


         # logging.info("Loading best model")
   #context.model.load(os.path.join( args.model_save_path,timestamp+ args.save_prefix +"_best_model"))
   #if context.data_type == DataType.SEQUENCE:
    #    datatools.sequence_classification.write_evaulation_report(context, context.val_loader,os.path.join(args.timestamp + report_path,args.save_prefix +".txt") , category_names=context.category_names) 
def get_dims_from_dataset(dataset_for_classification):
    if dataset_for_classification == "cifar_challenge" or dataset_for_classification == "cifar10" or dataset_for_classification =="minicifar10":
           img_h=32
           img_w=32
           channels=3
    elif dataset_for_classification == "mnist":
        img_h=28
        img_w=28
        channels=1
    else:
        raise Exception("dataset not supported for count_multiplies")
    return img_h, img_w, channels


def enact_adaptive_pruning(context, args, percentage, pruning_func, loader=None,  subblocks=True, reps=None):
    '''
    carry out adaptive pruning by calling by_block accuracies reps times
    if reps is None, then calls by_block accuracies once, then calls it a number of times equal to the number of items in by_block accuracies -1
    '''
    if reps == None:
        initial_dict, prunable_count= by_block_accuracies(context=context, args=args, percentage=percentage, pruning_func=pruning_func, loader=loader, enact=True, subblocks=subblocks)
        reps= max(prunable_count-1,0)
    for i in range(reps):
        by_block_accuracies(context=context, args=args, percentage=percentage,pruning_func=pruning_func, loader=loader, enact=True, subblocks=subblocks)
    logging.info("Current block prune counts: {}".format(block_prune_counts))


block_prune_counts = collections.Counter() #for counting which blocks are pruned by the adaptive method
def by_block_accuracies(context,args, percentage, pruning_func, loader=None, enact=False, subblocks=False) :
    #setting enact=True will cause the function actually enact the pruning operation that yields the smallest decrease in accuracy
    #pruning func should return True if the pruning operation suceeeded and False otherwise
    import candle.proxy
    if loader == None:
        loader = context.val_loader
    blocks = context.model.to_subblocks() if subblocks else context.model.to_blocks() 
    block_accuracies = collections.OrderedDict()
    accuracy_blocks = {}
    preprune_dict=copy.deepcopy(context.model.state_dict())
    #context.model.save("./temp/tempmodel")
    prunable_count=0 #number of potentially prunable blocks
    def doprune(targ_block):
        nonlocal prunable_count
        can_prune = False
        if isinstance(targ_block, candle.proxy.ProxyLayer):
            prunable_count+=1
            can_prune = pruning_func(targ_block)
        elif  getattr(targ_block,"apply_to_subproxies",None) is not None:
            prunable_count+=1
            success_list= targ_block.apply_to_subproxies(pruning_func)
            can_prune = all(success_list) 
        return can_prune

    for name, block in blocks.items():
        can_prune = doprune(block)
        if can_prune:
            acc=  basic_classification.evaluate(context, loader, no_grad=args.use_nograd)
            block_accuracies[name]=acc
            accuracy_blocks[acc]=name
            logging.debug("pruing{} yields accuracy of {}".format(name,block_accuracies[name]))
            context.model.load_state_dict(copy.deepcopy(preprune_dict)) #context.model.load("./temp/tempmodel" ) #reset model
    if enact:
        if len(accuracy_blocks.keys()) != 0: 
            best_acc =max(accuracy_blocks.keys())
            name_of_best=accuracy_blocks[best_acc]
            logging.debug("decided to prune:{}".format(name_of_best))
            block_prune_counts[name_of_best]+=1
            doprune(blocks[name_of_best] )

    return block_accuracies,prunable_count


def get_pruning_func(context, args):
  if args.sense_adaptive_pruning:
        def pfunc(prune_unit):
            logging.info("using_adaptive_pruing")
            one_layer_prune_func = get_one_layer_pruning_func(context, args, prune_unit)
            enact_adaptive_pruning(context, args, prune_unit, one_layer_prune_func,  subblocks=args.sense_adaptive_use_subblocks)
 
        return pfunc
  else: 

    if args.prune_layer_mode == "by_layer":
        assert args.proxy_context_type != "l1reg_context_slimming" 
        if args.group_prune_strategy == "random":
            logging.info("using random channel pruning")
            return functools.partial(context.model.proxy_ctx.prune,  method = "random")
        elif args.group_prune_strategy == "taylor": 
            logging.info("using taylor channel pruning")
            return functools.partial(context.model.proxy_ctx.prune,  method = "taylor")
        elif args.group_prune_strategy=="standard":
            logging.info("using  weight-norm pruning")
            return context.model.proxy_ctx.prune
        else:
            raise Exception("cannot deterimine correct pruning function")
    elif args.prune_layer_mode == "global":
        if args.global_prune_normalization =="no_normalization":
            normalize=False
            norm_mode =squeezenet.PruningNormalizationMode.NO_NORMALIZATION
        elif args.global_prune_normalization == "by_layer":
            normalize=True
            norm_mode = squeezenet.PruningNormalizationMode.BY_LAYER
        elif args.global_prune_normalization == "by_block":
            logging.info("using by_block normalization")
            normalize=True
            norm_mode = squeezenet.PruningNormalizationMode.BY_BLOCK
        else:
            raise Exception("Unknown normalization mode")

           

        if args.proxy_context_type == "l1reg_context_slimming" :
                return functools.partial(context.model.proxy_ctx.prune_global_smallest, mask_type=candle.prune.BatchNorm2DMask)
        elif args.group_prune_strategy == "taylor": 
                logging.info("using global taylor pruning")
                def do_global_taylor_prune(*pargs,**kwargs):
                    if args.global_prune_flop_regularize:
                        img_h, img_w, channels = get_dims_from_dataset(args.dataset_for_classification)
                        context.model.multiplies(img_h, img_w, channels, unpruned=False) #to record multiplies per layer
                        context.model.compute_flop_reg_terms_standard( img_h, img_w, channels)
                    context.model.compute_pruning_normalization_factor(norm_mode)
                    context.model.proxy_ctx.prune_global_smallest(*pargs,method="taylor",normalize=normalize, absolute=args.prune_absolute ,flop_reg = args.global_prune_flop_regularize, flop_reg_lambda=args.global_prune_flop_lambda, **kwargs)
         

                return do_global_taylor_prune  #functools.partial(context.model.proxy_ctx.prune_global_smallest, method="taylor")
        elif args.group_prune_strategy == "random":  
                logging.info("using global random pruning")
                def do_global_random_prune(*pargs,**kwargs):
                    if args.global_prune_flop_regularize:
                        img_h, img_w, channels = get_dims_from_dataset(args.dataset_for_classification)
                        context.model.multiplies(img_h, img_w, channels, unpruned=False) #to record multiplies per layer

                    context.model.proxy_ctx.prune_global_smallest(*pargs,method="random",normalize=normalize, absolute=args.prune_absolute, flop_reg = args.global_prune_flop_regularize, flop_reg_lambda=args.global_prune_flop_lambda   ,**kwargs)
                return do_global_random_prune
        elif args.group_prune_strategy == "normalized_channel":
            logging.info("using global normalized channel norm pruning")
            def do_global_normalized_channel(*pargs,**kwargs):
                    assert normalize == False #we use a different normalization implementation
                    context.model.proxy_ctx.prune_global_smallest(*pargs,method="normalized_channel_norm",normalize=False, absolute=args.prune_absolute, flop_reg = args.global_prune_flop_regularize, flop_reg_lambda=args.global_prune_flop_lambda   ,**kwargs)
            return do_global_normalized_channel

        else:
                raise Exception("cannot deterimine correct pruning function")

    else:
        raise Exception("Cannot determine correct pruning function")

def get_one_layer_pruning_func(context, args, prune_unit):
    import candle.proxy
    if args.prune_layer_mode == "by_layer":
        assert args.proxy_context_type != "l1reg_context_slimming" 
        if args.group_prune_strategy == "random":
            logging.info("using layer_targeted random channel pruning. ")
            return functools.partial(context.model.proxy_ctx.prune_proxy_layer, method="random",percentage=prune_unit, provider_type =candle.proxy.ProxyDecorator  )
        elif args.group_prune_strategy == "taylor":
            logging.info("using taylor channel pruning")
            return functools.partial(context.model.proxy_ctx.prune_proxy_layer, method="taylor",percentage=prune_unit, provider_type =candle.proxy.ProxyDecorator  )
        else:
            logging.info("using layer_targeted channel-based weight pruning") 
            return functools.partial(context.model.proxy_ctx.prune_proxy_layer, percentage=prune_unit, provider_type =candle.proxy.ProxyDecorator  )
    elif args.prune_layer_mode == "global":
            raise Exception("not implemented")
            assert args.proxy_context_type == "l1reg_context_slimming" 
    else:
        raise Exception("Cannot determine correct pruning function")




def hz_lasso_whole_model(context,args,num_samples, target_prop, loader,solve_for_weights):
    #create a copy of the model
    logging.info("initiating hz_lasso pruning")
    model_copy = copy.deepcopy(context.model)
    subblocks = context.model.to_subblocks()
    subblocks_copy = model_copy.to_subblocks()
    for sb_name in subblocks.keys():
        sb_real = subblocks[sb_name]
        sb_copy = subblocks_copy[sb_name]
        if not isinstance(sb_real, candle.proxy.ProxyConv2d):
            continue
        logging.info("pruning {}".format(sb_name))
        sb_real.store_input= True
        sb_copy.store_output = True
        for i,(batch_in, *other) in enumerate(loader): 
             with torch.no_grad():
                context.model(batch_in)
                model_copy(batch_in)
             if i >= num_samples-1:
                break
        context.model.proxy_ctx.hz_lasso_prune(sb_real, target_num_channels=None,target_prop=target_prop, sample_inputs=sb_real.record_of_input, sample_outputs=sb_copy.record_of_output, solve_for_weights=solve_for_weights)
        sb_real.store_input=False
        sb_copy.store_output=False

        sb_real.record_of_input=[]
        sb_copy.record_of_output=[]

def recalc_weights_pruned(context, args, num_samples, loader,model_copy):
    context.model.eval()
    logging.info("re-calculating weights")
    subblocks = context.model.to_subblocks()
    subblocks_copy = model_copy.to_subblocks()
    from tqdm import tqdm
    sb_loader =tqdm(subblocks.keys())
    for sb_name in sb_loader:
        sb_real = subblocks[sb_name]
        sb_copy = subblocks_copy[sb_name]
        if not isinstance(sb_real, candle.proxy.ProxyConv2d):
            continue
        logging.info("pruning {}".format(sb_name))
        sb_real.store_input= True
        sb_copy.store_output = True
        for i,(batch_in, *other) in enumerate(loader): 
             with torch.no_grad():
                context.model(batch_in)
                model_copy(batch_in)
             if i >= num_samples-1:
                break
        context.model.proxy_ctx.recalc_weights_pruned_layer(sb_real,Atensor=torch.cat(sb_real.record_of_input,dim=0), Ytensor=torch.cat(sb_copy.record_of_output,dim=0))
        sb_real.store_input=False
        sb_copy.store_output=False

        sb_real.record_of_input=[]
        sb_copy.record_of_output=[]
    context.model.train()






def enable_grad_storage(model):
   subblocks = model.to_subblocks()
   for name, layer in subblocks.items():
       if not isinstance(layer, candle.proxy.ProxyConv2d):
           continue
       layer.store_output=True
       layer.store_output_grad=True

def disable_grad_storage(model):
   subblocks = model.to_subblocks()
   for name, layer in subblocks.items():
       if not isinstance(layer, candle.proxy.ProxyConv2d):
           continue
       layer.store_output=False
       layer.store_output_grad=False




def update_abs_deriv_sum(model):
    subblocks = model.to_subblocks()
    for _, layer in subblocks.items():
                if not isinstance(layer, candle.proxy.ProxyConv2d):
                    continue
                layer.update_abs_deriv_sum()

def clear_abs_deriv_sum(model):
    subblocks = model.to_subblocks()
    for _, layer in subblocks.items():
                if not isinstance(layer, candle.proxy.ProxyConv2d):
                    continue
                layer.record_of_abs_deriv_sum=0

def clear_record_of_output(model):
    subblocks = model.to_subblocks()
    for _, layer in subblocks.items():
                if not isinstance(layer, candle.proxy.ProxyConv2d):
                    continue
                layer.record_of_output=[]




def taylor_sample_batches(context, args):
    #note: this function may change a model slightly by changing its batch norm running averages
   assert args.group_prune_strategy == "taylor" 
   context.model.eval()
   loader=context.val_loader #using a subset of the train set will cause randomness because the train set is shuffled
   subblocks = context.model.to_subblocks()
   for name, layer in subblocks.items():
       if not isinstance(layer, candle.proxy.ProxyConv2d):
           continue
       layer.store_output=True
       layer.store_output_grad=True
   for i, (batch_in,*other) in enumerate(loader):
            categories = other[0]
            scores = context.model(batch_in)
            loss=  F.cross_entropy(scores,categories) 
            loss.backward()
            for _, layer in subblocks.items():
                if not isinstance(layer, candle.proxy.ProxyConv2d):
                    continue
                layer.update_abs_deriv_sum()

            context.optimizer.zero_grad()
            #if i >= args.taylor_num_samples -1:
            #    break
   context.model.train()
            
            

def taylor_sample_clear(context,args):
   subblocks = context.model.to_subblocks()
   for name, layer in subblocks.items():
       if not isinstance(layer, candle.proxy.ProxyConv2d):
           continue
       layer.store_output=False
       layer.record_of_output=[]
       layer.record_of_output_grad=[]

