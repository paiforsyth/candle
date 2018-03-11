import torch.nn.functional as F
import torch
def evaluate(context): 
   correct=0
   total=0
   context.model.eval()
   for seqs, categories, pad_mat in context.val_loader:
        total+=seqs.shape[0]
        scores=context.model(seqs,pad_mat)
        scores=F.softmax(scores,dim=1)
        _,predictions=torch.max(scores,dim=1)
        correct+= torch.sum(predictions==categories).cpu().data[0]
   context.model.train()
   return correct / total 
