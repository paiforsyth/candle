import logging
import numpy as np
import math
import os
import torch
import torch.nn as nn
PRINT_INTERVAL = 20000



def fasttext_from_file(configure,  wordindexer):
        path=os.path.join(configure.fasttext_path,'wiki.en.vec')
        logging.info("Loading word vectors from file: " + path)
        index2vec ={}
        local_word_set =  set(wordindexer.word2index.keys()).copy()
        n_found = 0
        with open(path) as f:
                # based on https://github.com/facebookresearch/MUSE/blob/master/src/utils.py
                for i, line in enumerate(f):
                    if i == 0:
                        dimension = int(line.split()[1])
                        continue
                    if  not local_word_set:  #the local word set is empty
                        break
                    if i % PRINT_INTERVAL == 0:
                        logging.debug("Read " + str(i) + " lines so far.  Found " +str(n_found) + " word vectors.")
                        logging.debug("Missing " + str(len(local_word_set)) + " word vectors.")
                    word, vec = line.rstrip().split(' ', 1)

                    if word not in local_word_set:
                        continue
                    else:
                        local_word_set.remove(word)
                    vec = np.fromstring(vec, sep=' ')
                    if np.linalg.norm(vec) == 0:
                        vec[0] = 0.01
                    index2vec[ wordindexer.word2index[word]  ] = torch.Tensor(vec)
                    n_found += 1
        logging.info("Loaded " + str(n_found) + " word vectors from file")

        if local_word_set:
            logging.info("Did not find word vectors for " +str(len(local_word_set)) +" words.")
        return index2vec, local_word_set


def embedding(index2vec,n_words, dim):
    emb=nn.Embedding(n_words,dim)
    for index in index2vec.keys():
        emb.weight.data[index,:] = index2vec[index]
    return emb
    
