import torch.utils.data as data
import os
import logging
import torch

from . import wordindexer
from  . import word_vectors
from  . import text_tool
from  . import sequence_classification



def load(args):
    sfname=args.save_prefix
    if args.use_saved_processed_data and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_traindataset')) and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_valdataset'))   and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_indexer')) and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_vects')):
        logging.info("Found cached "+sfname+" data.  Loading it.")
        train_dataset = torch.load( os.path.join(args.processed_data_path,sfname+'_traindataset')  )
        val_dataset = torch.load( os.path.join(args.processed_data_path,sfname+'_valdataset')  )
        vects = torch.load( os.path.join(args.processed_data_path,sfname+'_vects')  )
        indexer = torch.load( os.path.join(args.processed_data_path,sfname+'_indexer')  )
    else:
        logging.info("Could not find cached "+sfname  +" data. Indexing words.")
        indexer=wordindexer.WordIndexer()

        def index_words(filename):
            f= open( os.path.join(args.ds_path,filename), errors= 'surrogateescape' )
            for line in f:
                    sentence=line.strip()
                    sentence = text_tool.normalize_string(sentence) 
                    indexer.add_sentence(sentence)
            f.close()
    
        index_words("rt-polarity.neg")
        index_words("rt-polarity.pos")
        indexer= indexer.trimmed(args.data_trim)
        vects, missing_words = word_vectors.fasttext_from_file(args, indexer) 
        sequences=[]
        categories=[]
        raw_sentences=[]
        def get_sentences_and_classes(filename,category):
            with open( os.path.join(args.ds_path,filename),errors= 'surrogateescape' ) as f:
                for line in f:
                    sentence=line.strip()
                    norm_sentence=text_tool.normalize_string(sentence)
                    seq=indexer.sentence2seq(norm_sentence,include_sos_eos=True) 
                    sequences.append(seq)
                    categories.append(category)
                    raw_sentences.append(sentence)

        get_sentences_and_classes("rt-polarity.neg",0)
        get_sentences_and_classes("rt-polarity.pos",1)
       
        combined_dataset=sequence_classification.Dataset(sequences,categories,raw_sequences = raw_sentences)
        combined_dataset.shuffle()
        val_dataset, train_dataset= combined_dataset.split(args.validation_set_size)
        train_dataset.remove_raw() #we dont need the raw sentences in the training set
        
        torch.save(train_dataset, os.path.join(args.processed_data_path,sfname+'_traindataset') )
        torch.save(val_dataset, os.path.join(args.processed_data_path,sfname+'_valdataset') )
        torch.save(vects, os.path.join(args.processed_data_path,sfname+'_vects') )
        torch.save(indexer, os.path.join(args.processed_data_path,sfname+'_indexer') )
    return train_dataset, val_dataset, vects, indexer



