import collections
import logging
import pickle

import torch.optim as optim

import candle.context

from . import models
from . import datatools.img_tools
from . import datatools.set_cifar_challenge
from . import datatools.basic_classification
def add_primary_args(parser):
    parser.add_argument("--dataset_for_classification",type=str,choices=[ "mnist", "cifar"],default="simple")
    parser.add_argument("--model",type=str, choices= ["shufflenet"], default="shufflenet")
    parser.add_argument("--proxy_mode", type=str, choices=["identity"], default="identity")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd"])
    parser.add_argument("--init_lr",type=float, default=0.1)
    parser.add_argument("--sgd_momentum",type=float, default=0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0)


Environment = collections.NamedTuple("Environment", "model, ctx, optimizer, val_loader, train_loader, tb_writer")
def make_environment(args):
    if args.proxy_mode == "identity":
        ctx = candle.context.Context()
    else:
        raise Exception("unknown proxy mode")
    if args.model == "shufflenet":
        model = models.ShuffleNet.from_args(ctx, args)
    else:
        raise Exception("unknown model")
    if args.use_cuda:
        model = model.cuda()

    if args.optimizer = "sgd":
        optimizer=optim.SGD(model.parameters(),lr=args.init_lr, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay ) 

    if args.dataset_for_classification == "cifar":
        f=open("../data/cifar/train_data","rb")
        squashed_images=pickle.load(f)
        labels=pickle.load(f)
        f.close()
        train_dataset,val_dataset = dtatools.set_cifar_challenge.make_train_val_datasets(squashed_images, labels, args.validation_set_size, transform=None, shuf=args.cifar_shuffle_val_set) 
        tr = transforms.Compose([transforms.RandomCrop(size=32 ,padding= 4), transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
        if args.cifar_random_erase:
                tr=transforms.Compose([tr, datatools.img_tools.RandomErase()])
        train_dataset.transform = tr
        val_dataset.transform = transforms.ToTensor()
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=datatools.basic_classification.make_var_wrap_collater(args))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=datatools.basic_classification.make_var_wrap_collater(args))
   
       
    return Environment(model=model, ctx=ctx, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader)



def train(args):
    enviro = make_environment(args)
    model_params = ctx.list_model_params()
    logging.info("Unpruned parameters: {}".format(ctx.count_unpruned()))
    for epoch_count in range(args.num_epochs):
        for batch_in, categories in context.train_loader: 
            enviro.optimizer.zero_grad()
            scores = enviro.model.batch_in 
            loss =  F.cross_entropy(scores,categories) 
            loss.backward()
            enviro.optimizer.step()



def main():
    parser = argparse.ArgumentParser()
    add_primary_args(parser)
    models.add_shufflenet_args(parser)
    args = parser.parse_args()
    train(args)


    # train_binary(args)
    train_pruned(args)

if __name__ == "__main__": 
    main()
