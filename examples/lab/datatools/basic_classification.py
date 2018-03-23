#test
from enum import Enum
from contextlib import suppress
from tqdm import tqdm
import logging
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.autograd import Variable
class DataType(Enum):
    SEQUENCE=1
    IMAGE=2

def evaluate(context, loader, no_grad =False): 
   correct=0
   total=0
   context.model.eval()
   context_manager = torch.no_grad() if no_grad else  suppress()
 #  with torch.no_grad():
   with context_manager:
    for batch, *other in loader:
         categories=other[0]
         if context.data_type==DataType.SEQUENCE:
             pad_mat = other[1]
         total+=batch.shape[0]
         scores= context.model(batch,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch)  #should have dimension batchsize
         scores=F.softmax(scores,dim=1)
         _,predictions=torch.max(scores,dim=1)
         if predictions.is_cuda:
            categories=categories.cuda(predictions.get_device())
         correct+= float(torch.sum(predictions==categories).float().cpu())
 #  context.model.train()
   return correct / total 



def predict(context, loader, no_grad=False): 
   '''
   '''
   context_manager = torch.no_grad() if no_grad else  suppress()
   context.model.eval()
   with context_manager:
    overall_predictions=[]
    logging.info("Predicting.")
    for batch, *other in loader:
         categories=other[0]
         if context.data_type==DataType.SEQUENCE:
             pad_mat = other[1]
         scores= context.model(batch,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch)  #should have dimension batchsize by number of categories
         scores=F.softmax(scores,dim=1)
         _,predictions_this_batch=torch.max(scores,dim=1)
         overall_predictions.extend(predictions_this_batch.data.tolist())
    #context.model.train()
   return overall_predictions 


def optimize_ensemble_on_val(contexts,val_loader):
   overall_predictions=[]
   logging.info("Predicting val.")
   score_list_2d=[]
   num_models= len(contexts)
   for i in range(len(val_loader)):
        score_list_2d.append([])
   for context in contexts:
       context.unstash_model()
       context.model.eval()
       for i, (batch, *other) in tqdm(enumerate(val_loader)):
            if context.data_type==DataType.SEQUENCE:
                pad_mat = other[1]

            scores= context.model(batch,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch)  #should have dimension batchsize by number of categories
            scores=F.log_softmax(scores,dim=1)
            scores=scores.unsqueeze(2)
            score_list_2d[i].append(scores.data)
    #   context.model.train()
       context.stash_model()

   category_list=[]
   for batch, *other in val_loader:
            categories = other[0]
            category_list.append(categories.data)
   category_tensor=torch.cat(category_list, dim=0)#dimension datasetsize
    
   batch_scores=[]
   for i in range(len(val_loader)):
        batch_scores.append(torch.cat(score_list_2d[i],dim=2))# result of this cat operation has dimension batchsize by num categories by num models
   score_tensor=torch.cat(batch_scores, dim=0) #has dimensions number of datapoints by num catergories by num models
   meta_model=nn.Linear(num_models, 1)    
   meta_model.weight.data.fill_(1)
   score_variable=Variable(score_tensor)
   category_variable=Variable(category_tensor) 
   if torch.cuda.is_available():
        meta_model=meta_model.cuda()
        score_variable=score_variable.cuda()
        category_varaible = category_variable.cuda()

   def eval_linear_model(model):
       combined_scores=model(score_variable).squeeze(2)
       _,predictions=torch.max(combined_scores, dim=1)
       acc= sum(( predictions.cpu() == category_variable.cpu() ).float()  )/len(predictions.data.tolist())
       return acc.data[0]
   logging.info("Equal-weight validation accuracy= "+str(eval_linear_model(meta_model)))
   optimizer=torch.optim.Adam(meta_model.parameters(),lr=0.01)
   NUM_ITER=2000 
   logging.info("optimally combining predictors")
   for i in tqdm(range(NUM_ITER)):
       y = meta_model(score_variable).squeeze(2)#should have dimension datapoints by categories
       meta_loss = F.cross_entropy(y,category_variable)
       optimizer.zero_grad()
       meta_loss.backward()
       optimizer.step()
   logging.info("Optimized validation accuracy= "+str(eval_linear_model(meta_model)))
   return meta_model

    

def ensemble_predict(contexts, loader, meta_model=None):
   '''
   Note: models are expected to be stashed
   args:
    -meta_model is a nn.Linear for combining scores of different modelss
    
   '''
   if meta_model is not None:
       logging.info("using provided meta_model")
   overall_predictions=[]
   logging.info("Predicting.")
   score_list_2d=[]
   for i in range(len(loader)):
        score_list_2d.append([])
   for context in contexts:
       context.unstash_model()
       context.model.eval()
       for i, (batch, *other) in tqdm(enumerate(loader)):
            categories=other[0]
            if context.data_type==DataType.SEQUENCE:
                pad_mat = other[1]

            #with torch.no_grad():
            scores= context.model(batch,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch)  #should have dimension batchsize by number of categories
            scores=F.log_softmax(scores,dim=1)
            scores=scores.unsqueeze(2)
            score_list_2d[i].append(scores.data)
    #   context.model.train()
       context.stash_model()
    
   batch_mean_scores=[]
   for i in range(len(loader)):
       if meta_model is None:
           batch_mean_scores.append(torch.mean(torch.cat(score_list_2d[i],dim=2),dim=2 ))
       else: 
           curscores= Variable(torch.cat(score_list_2d[i],dim=2))
           if torch.cuda.is_available():
                curscores=curscores.cuda()
           batch_mean_scores.append(meta_model( curscores ).squeeze(2).data  )
   batch_mean_score_tensor=torch.cat(batch_mean_scores, dim=0)
   _,predictions=torch.max(batch_mean_score_tensor,dim=1)
   return predictions.tolist() 


def make_prediction_report(context, loader, filename,no_grad):
    f=open(filename,"w")
    f.write("ids,labels\n")
    predictions = predict(context, loader, no_grad)
    index=0
    for index, prediction in enumerate(predictions):
        f.write(str(index)+","+str(prediction) + "\n")
    f.close()

def make_ensemble_prediction_report(contexts, loader, filename, meta_model=None):
    predictions = ensemble_predict(contexts, loader, meta_model)
    f=open(filename,"w")
    f.write("ids,labels\n")
    index=0
    for index, prediction in enumerate(predictions):
        f.write(str(index)+","+str(prediction) + "\n")
    f.close()

def score_report(filename,ground_truth):
    f = open(filename,"r")
    correct = 0
    for index,line in enumerate(f):
        if index == 0: #first line is header
            continue
        if int(line.strip().split(",")[1])  == ground_truth[index-1]: #count of report is 1 ahead of ground truth, since report has a header
            correct += 1 
        
    f.close()  
    return float(correct) / float(index) #there are index+1 lines in the file.  However, line 0 is a header

def make_var_wrap_collater(args, volatile= False ):
    def collater(batch_in):
       batch_in, categories, *rest=torch.utils.data.dataloader.default_collate(batch_in)
       batch_in = Variable(batch_in, volatile=volatile)
       categories =Variable(categories)
       if args.cuda:
           batch_in = batch_in.cuda()
           categories = categories.cuda()
       return tuple([batch_in, categories, *rest])
    return collater
