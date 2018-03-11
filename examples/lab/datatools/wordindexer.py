import logging
UNKNOWN_TOKEN=3
NULL_TOKEN=0
class WordIndexer:

    def __init__(self, version = "std" ):
        self.version=version
        if version =="std":
            self.word2index = {}
            self.word2count = {}
            self.index2word = ["NULL","SOS", "EOS", "UNKNOWN"]
            self.n_words = 4
            self.sos_tokens={1}
            self.eos_tokens={2}
            self.micl_tokens={0,3}
        else:
            raise Exception("Unknown WordIndexer Version")

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # def remove_word(self, word):
        # assert word not in self.sos_tokens and word not in self.eos_tokens and word not in self.micl_tokens
        # index=self.word2index[word]
        # del self.word2index[word]
        # del self.word2count[word]
        # del self.index2word[index]
        # self.n_words-=1

    def trimmed(self,num_to_keep):
        '''
            Creates a new wordindexer by trimming this one.  Note that in general this will not preserve any indexing
        '''
        logging.info("Trimming word index.")
        words=  list(self.word2count.keys())
        words.sort(key= lambda word: self.word2count[word])
        to_remove=set(words[:-num_to_keep])
        
        trimmed_index=WordIndexer(self.version)
        
        for word in self.word2count:
            if word in to_remove:
                continue
            trimmed_index.add_word(word)
        
        
        logging.info("After trimming, index has "+ str(trimmed_index.n_words)+ " words.")
        return trimmed_index
         

    def seq2sentence(self, dexes):
        '''
         Converts a list of indicies into the corresponding string
         Removes SOS and EOS tokens, but not other special tokens
         '''
       # if dexes[0] not in self.sos_tokens:
        #    import pdb; pdb.set_trace()
        assert dexes[0] in self.sos_tokens 
        str = ""
        for count, index in enumerate(dexes):
            if count == 0: 
                continue
            if index in self.eos_tokens:
                return str.strip()
            str += self.index2word[index]
            str += " "
        raise Exception("No EOS token found")

     

    def sentence2seq(self, sentence, include_sos_eos=True, sos=1,eos=2):
        '''
            returns an index representation of the word, optionally including including SOS and EOS tokens
        '''
        assert sos in self.sos_tokens
        assert eos in self.eos_tokens
        dexes = [self.word2index.get(word,UNKNOWN_TOKEN) for word in sentence.split()]
        if include_sos_eos:
            dexes.insert(0, sos)
            dexes.append(eos)
        return dexes


