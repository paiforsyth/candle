import random
import torch.utils.data as data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class Dataset(data.Dataset):
    '''
    raw_sequences is an optional item that stores an unprocessed sequences. used mainly in evaluation
    '''
    def __init__(self, sequences, categories, raw_sequences=None):
        assert len(sequences) == len(categories)
        assert len(sequences) > 0
        self.sequences = sequences
        self.categories = categories
        if raw_sequences is None:
            self.raw_sequences= [None]*len(sequences)
        else:
            self.raw_sequences=raw_sequences
        
    def __getitem__(self,idx):
            return (self.sequences[idx], self.categories[idx], self.raw_sequences[idx] )

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        c = list(zip(self.sequences, self.categories, self.raw_sequences))
        random.shuffle(c)
        self.sequences, self.categories, self.raw_sequences = zip(*c)

    def split(self, index):
        return Dataset(self.sequences[:index],self.categories[:index], self.raw_sequences[:index]), Dataset(self.sequences[index:],self.categories[index:], self.raw_sequences[index:] )

    def remove_raw(self):
        self.raw_sequences=[None]*len(self.sequences)

def make_collater(args):
    def collater(batch):
        seqs=[entry[0] for entry in batch ]
        categories=[entry[1] for entry in batch]
        lengths=[len(entry[0]) for entry in  batch  ]
        raw_seqs=[entry[2] for entry in batch ]
        max_len=max(lengths)
        pt_seqs=torch.LongTensor(len(seqs),max_len).zero_()
        pad_mat=torch.ones(len(seqs),max_len)
        for i in range(len(seqs)):
            pt_seqs[i,:lengths[i]]= torch.LongTensor(seqs[i])
            pad_mat[i,:lengths[i]]=0
        pt_seqs=Variable(pt_seqs)
        pt_categories= Variable(torch.LongTensor(categories))
        pad_mat=Variable(pad_mat)
        pad_mat=pad_mat.view(-1,max_len,1)
        if args.cuda:
            pt_seqs=pt_seqs.cuda()
            pt_categories=pt_categories.cuda()
            pad_mat=pad_mat.cuda()
        return pt_seqs, pt_categories, pad_mat, raw_seqs
    return collater


   
def evaluation_report(context, loader, category_names={}, divider= "\t"):
    context.model.eval()
    correct=0
    total=0
    s="Correctness\tOriginal Sequence\tCoded Sequence\tReconstructed Sequence\tPredicted Category\tTrue Category\n"
    for seqs, categories, pad_mat, raw_sequences in loader:
        for sequence, category, pad_mat_row, raw_sequence in zip(seqs, categories, pad_mat, raw_sequences):
               #import pdb; pdb.set_trace()
               scores=context.model(sequence.view(1,-1),pad_mat_row.view(1,-1,1))
               _,prediction=torch.max(scores,dim=1)
               correct_this_seq=prediction.cpu().data[0] == category.cpu().data[0]
               
               s+= "CORRECT\t" if correct_this_seq else "INCORRECT\t"

               s+= raw_sequence.strip() +"\t"

               s+= " ".join([str(item) for item in sequence.data.tolist()])
               s += "\t"
               
               s+= context.indexer.seq2sentence(sequence.data.tolist()) +"\t"

               pred = prediction.cpu().data[0]
               s+=  str(category_names.get(pred,pred))+"\t"
                
               caty=category.cpu().data[0]
               s+= str( category_names.get(caty,caty))
               s+="\n"

               total+=1
               if correct_this_seq:
                   correct+=1
    accuracy=correct / total
    s= "Accuracy: "+ str(accuracy)+"\n"+s
    context.model.train()
    return s, accuracy

def write_evaulation_report(context,loader,filename,category_names):
    s,_=evaluation_report(context, loader, category_names)
    f=open(filename,"w",errors='surrogateescape')
    f.write(s)
    f.close()
                
