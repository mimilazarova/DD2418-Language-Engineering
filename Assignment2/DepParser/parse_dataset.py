import numpy as np
from collections import defaultdict


class Dataset() :
    def __init__(self) :
        # Words that appear less than THRESHOLD times will be translated
        # into <UNK>
        self.THRESHOLD = 1000
        self.wordcount = defaultdict(int)
        self.datapoints = []
        # Mappings from words/postags to indices
        self.w2i = { "<UNK>":0, "<EOS>":1, "<EMPTY>":2 }
        self.p2i = { "<EOS>":0, "<EMPTY>":1 }
        self.__dim = None
        self.__number_of_words = None
        self.__number_of_postags = None

    def copy_feature_maps(self, other):
        self.w2i = other.w2i
        self.p2i = other.p2i
        self.THRESHOLD = other.THRESHOLD
        self.wordcount = other.wordcount

    def word2int(self, word) :
        if word not in self.w2i :
            self.w2i[word] = len(self.w2i)
        return self.w2i[word]
            
    def postag2int(self, tag) :
        if tag not in self.p2i :
            self.p2i[tag] = len(self.p2i)
        return self.p2i[tag]
    
    def increment_word_count(self, word) :
        self.wordcount[word] += 1
        
    def number_of_words_above_threshold(self) :
        n=0
        for w in self.wordcount.keys() :
            if self.wordcount[w] > self.THRESHOLD :
                n += 1
        return n

    def get_features(self, words, tags, i, stack, train=False):
        """
        A datapoint is represented by means of 6 features:
          - w1, the next word in the buffer 
          - t1, the POS tag of the next word in the buffer
          - w2, the topmost word on the stack
          - t2, the POS tag of the topmost word on the stack
          - w3, the second-topmost word on the stack
          - t3, the POS tag of the second-topmost word on the stack
        """

        w1 = words[i] if i < len(words) else "<EOS>"
        t1 = tags[i] if i < len(tags) else "<EOS>"
        if len(stack) >= 1:
            s1 = stack[-1]
            w2 = words[s1]
            t2 = tags[s1]
        else:
            w2 = "<EMPTY>"
            t2 = "<EMPTY>"
            
        if len(stack) >= 2:
            s2 = stack[-2]
            w3 = words[s2]
            t3 = tags[s2]
        else:
            w3 = "<EMPTY>"
            t3 = "<EMPTY>"
        
        self.postag2int(t1)
        self.postag2int(t2)    
        self.postag2int(t2)

        if train and self.THRESHOLD:
            self.increment_word_count(w1)
            self.increment_word_count(w2)
            self.increment_word_count(w3)
        
        return [w1,t1,w2,t2,w3,t3] if self.THRESHOLD else [None,t1,None,t2,None,t3]

    def add_datapoint(self, words, tags, i, stack, action, train=False) :
        self.datapoints.append(self.get_features(words, tags, i, stack, train) + [action])

    def features2array(self, features, as_indices=False):
        w1,t1,w2,t2,w3,t3 = features

        if self.THRESHOLD:
            # First change uncommon words to <UNK>
            if self.wordcount[w1] <= self.THRESHOLD :
                w1 = "<UNK>"
            if self.wordcount[w2] <= self.THRESHOLD :
                w2 = "<UNK>"
            if self.wordcount[w3] <= self.THRESHOLD :
                w3 = "<UNK>"
            
            word_features = [
                self.word2int(w1),
                self.word2int(w2) + self.number_of_words,
                self.word2int(w3) + self.number_of_words*2
            ]
        else:
            word_features = []

        pos_features = [
            self.postag2int(t1) + self.number_of_words*3,
            self.postag2int(t2) + self.number_of_words*3 + self.number_of_postags,
            self.postag2int(t3) + self.number_of_words*3 + self.number_of_postags*2
        ]

        index = word_features + pos_features
        
        if as_indices:
            return index

        x = np.zeros(self.dim)
        for j in index :
            x[j] = 1.
        return x

    def dp2array(self, words, tags, i, stack):
        return self.features2array(self.get_features(words, tags, i, stack))

    @property
    def number_of_words(self):
        if self.__number_of_words is None:
            # the "+1" is for <UNK>
            self.__number_of_words = self.number_of_words_above_threshold() + 1 if self.THRESHOLD else 0
            print( "number of words = ", self.__number_of_words )
        return self.__number_of_words

    @property
    def number_of_postags(self):
        if self.__number_of_postags is None:
            self.__number_of_postags = len(self.p2i)
            print( "number of postags = ", self.__number_of_postags )
        return self.__number_of_postags
    

    @property
    def dim(self):
        if self.__dim is None:
            self.__dim = 3*self.number_of_words + 3*self.number_of_postags
            print( "dim = ", self.__dim )
        return self.__dim

    def to_arrays(self) :
        """ 
        Creates numpy arrays containing a numerical encoding of the data.
        Uncommon words are mapped to <UNK> in order to reduce the dimensionality.
        Considering we have 3 word features and 3 postag features, the dimensionality 
        will be:
        
           3 * (number of unique words + 1)  /* the +1 is for <UNK> */
         + 3 * (number of unique postags)

        """
        x = np.zeros((len(self.datapoints),self.dim))
        y = []
        for i in range(len(self.datapoints)) :
            for j in self.features2array(self.datapoints[i][:-1], as_indices=True):
                x[i][j] = 1.
            y.append( self.datapoints[i][-1] )
        y = np.array(y)
        return x, y
