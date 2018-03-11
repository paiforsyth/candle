import torch
from  . import text_tool
def load(args):
    indexer=wordindexer.WordIndexer()
    def index_words(filename):
        f= open( os.path.join(args.ds_path,filename) )
        for line in f:
                sentence=line
                sentence = text_tool.normalize_string(sentence) 
                indexer.add_sentence(sentence)
        f.close()
        index_words("train.en")    
        indexer= indexer.trimmed(args.data_trim)
        vects, missing_words = word_vectors.fasttext_from_file(args, indexer) 
        sequences = []
        raw_sequences = []
        def get_sentences(filename):
            with open( os.path.join(args.ds_path,filename) ) as f:
                for line in f:
                    sentence=line
                    norm_sentence=text_tool.normalize_string(sentence)
                    seq=indexer.sentence2seq(norm_sentence,include_sos_eos=True) 
                    sequences.append(seq)
                    raw_sequences.append(sentence)
        get_sentences("train.en")
        combined_dataset=sequence_to_sequence.Dataset(sequences,sequences,raw_sequences, raw_sequences) #autoencoding
        combined_dataset.shuffle()
        val_dataset, train_dataset= combined_dataset.split(args.validation_set_size)
        train_dataset.remove_raw() #we dont need the raw sentences in the training set

        torch.save(train_dataset, os.path.join(args.processed_data_path,sfname+'_traindataset') )
        torch.save(val_dataset, os.path.join(args.processed_data_path,sfname+'_valdataset') )
        torch.save(vects, os.path.join(args.processed_data_path,sfname+'_vects') )
        torch.save(indexer, os.path.join(args.processed_data_path,sfname+'_indexer') )
        return train_dataset, val_dataset, vects, indexer



