import argparse
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
from . import __main__ as mainfuncs
from  .modules import saveable_data_par 
from .genutil import modules
from torchvision import transforms
import torchvision.datasets as tvds
def add_args(parser):
    if parser is None:
        parser= argparse.ArgumentParser() 
    parser.add_argument("--dataset_for_classification",type=str,choices=["simple","moviepol", "mnist", "cifar_challenge", "cifar10"],default="simple")

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
       assert args.enable_l1_reg and args.proxy_context_type == "l1reg_context_slimming"
  
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
        #import pdb; pdb.set_trace()

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



   else:
        raise Exception("Unknown dataset.")
   


   logging.info("using save prefix "+str(args.save_prefix))


   if data_type == DataType.SEQUENCE:
        embedding=word_vectors.embedding(index2vec, indexer.n_words,300)
        train_loader= data.DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True,collate_fn = sequence_classification.make_collater(args))
        val_loader= data.DataLoader(val_dataset,batch_size = args.batch_size, shuffle = False, collate_fn = sequence_classification.make_collater(args))
   elif data_type == DataType.IMAGE:
       indexer= None
       if args.mode == "train":
            train_loader=data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=basic_classification.make_var_wrap_collater(args))
            val_loader=data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=basic_classification.make_var_wrap_collater(args,volatile=True ))
       elif  args.mode == "test":
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
   elif args.lr_scheduler == None:
       scheduler = None
   else: 
       raise Exception("Unknown Scheduler")


   if args.mode =="train":
       train_size=len(train_dataset)
       test_loader = None
   elif args.mode=="test":
       train_size= None
       train_loader = None


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
   if args.mode == "test":
        basic_classification.make_prediction_report(context, context.test_loader,args.test_report_filename, no_grad=args.use_nograd)  
        return
   
   if args.lr_scheduler == "epoch_anneal":
        epoch_anneal_cur_cycle=0
        
   
   context.tb_writer.write_hyperparams()
   timestamp=reporting.timestamp()
   
   report_interval=max(len(context.train_loader) //  args.reports_per_epoch ,1)
   accumulated_loss=0 
   param_count=genutil_modules.count_trainable_params(context.model)
   if args.proxy_context_type == "no_context": 
        param_count = modules.count_trainable_params(context.model)
   else:
        param_count = modules.count_elem(context.model.proxy_ctx.list_model_params() ) 

   logging.info("Number of parameters: "+ str(param_count))
   context.tb_writer.write_num_trainable_params(param_count)


   if args.prune_trained:
       context.model.proxy_ctx.prune(args.prune_trained_pct)
       n_unpruned = context.model.proxy_ctx.count_unpruned()
       logging.info("Unpruned masks: "+str(n_unpruned))
       context.model.save(os.path.join( args.model_save_path, args.res_file+"_prune_" + str(args.prune_trained_pct) )  )
       return

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
       if args.dataset_for_classification == "cifar_challenge" or args.dataset_for_classification =="cifar10":
           img_h=32
           img_w=32
           channels=3
       print("Approx number of multiplies: ", countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels))    
       return
   if args.enable_pruning:
        init_mask_count = context.model.proxy_ctx.count_unpruned()
        logging.info("Initial number of masks {}".format(init_mask_count))
        if args.prune_target_frac is not None:
            prune_target = int(init_mask_count *args.prune_target_frac )
        else:
            prune_target =args.prune_target
        logging.info("Target number of masks is :{}".format(prune_target))
   
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
           # import pdb; pdb.set_trace() 
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
                loss += context.model.proxy_ctx.l0_loss(args.l0reg_lambda) 

            if args.enable_l1reg:
                loss +=context.model.proxy_ctx.l1_loss_slimming(args.l1reg_lambda)


            loss.backward()


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
        eval_score=basic_classification.evaluate(context, context.val_loader,no_grad=args.use_nograd)
        context.tb_writer.write_accuracy(eval_score)
        logging.info("Finished epoch number "+ str(epoch_count+1) +  " of " +str(args.num_epochs)+".  Accuracy is "+ str(eval_score) +".")
        if args.report_unpruned:
            n_unpruned = float(context.model.proxy_ctx.count_unpruned())
            logging.info("Unpruned masks: "+str(n_unpruned))
            context.tb_writer.write_unpruned_params(n_unpruned)
           
        if args.enable_pruning: 
             if epoch_count >= args.prune_warmup_epochs and epoch_count % args.prune_epoch_freq==0 and n_unpruned> prune_target:
                context.model.proxy_ctx.prune(1)
       
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
                if args.dataset_for_classification == "cifar_challenge" or args.dataset_for_classification == "cifar10":
                    img_h=32
                    img_w=32
                    channels=3
                    mults = countmult.count_approx_multiplies(context.model, img_h=img_h, img_w=img_w, input_channels=channels)
                    logging.info("Approx number of multiplies: "+str(mults) )    
                    context.tb_writer.write_multiplies(mults)
                else:
                    raise Exception("dataset not supported for count_multiplies")

         # logging.info("Loading best model")
   #context.model.load(os.path.join( args.model_save_path,timestamp+ args.save_prefix +"_best_model"))
   #if context.data_type == DataType.SEQUENCE:
    #    datatools.sequence_classification.write_evaulation_report(context, context.val_loader,os.path.join(args.timestamp + report_path,args.save_prefix +".txt") , category_names=context.category_names) 
