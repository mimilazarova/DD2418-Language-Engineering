import numpy as np
from collections import defaultdict


class Dataset():

    def __init__(self):

        # Words that appear less than THRESHOLD times will be translated
        # into <UNK>
        self.THRESHOLD = 1000
        self.wordcount = defaultdict(int)
        self.datapoints = []
        # Mappings from words/postags to indices
        self.w2i = {"<UNK>": 0, "<EOS>": 1, "<EMPTY>": 2}
        self.p2i = {"<EOS>": 0, "<EMPTY>": 1}

    def word2int(self, word):
        if word not in self.w2i:
            self.w2i[word] = len(self.w2i)
        return self.w2i[word]

    def postag2int(self, tag):
        if tag not in self.p2i:
            self.p2i[tag] = len(self.p2i)
        return self.p2i[tag]

    def increment_word_count(self, word):
        self.wordcount[word] += 1

    def number_of_words_above_threshold(self):
        n = 0
        for w in self.wordcount.keys():
            if self.wordcount[w] > self.THRESHOLD:
                n += 1
        return n

    def add_datapoint(self, words, tags, i, stack, action):
        """
        A datapoint is represented by means of 6 features:
          - w1, the next word in the buffer 
          - t1, the POS tag of the next word in the buffer
          - w2, the topmost word on the stack
          - t2, the POS tag of the topmost word on the stack
          - w3, the second-topmost word on the stack
          - t3, the POS tag of the second-topmost word on the stack

        + the correct class y (one of the actions SH, LA, RA). 
        """
        w1 = words[i] if i < len(words) else "<EOS>"
        t1 = tags[i] if i < len(tags) else "<EOS>"
        self.increment_word_count(w1)
        self.postag2int(t1)
        if len(stack) >= 1:
            s1 = stack[-1]
            w2 = words[s1]
            t2 = tags[s1]
        else:
            w2 = "<EMPTY>"
            t2 = "<EMPTY>"
        self.increment_word_count(w2)
        self.postag2int(t2)
        if len(stack) >= 2:
            s2 = stack[-2]
            w3 = words[s2]
            t3 = tags[s2]
        else:
            w3 = "<EMPTY>"
            t3 = "<EMPTY>"
        self.increment_word_count(w3)
        self.postag2int(t2)
        self.datapoints.append((w1, t1, w2, t2, w3, t3, action))

    def dataset2arrays(self):
        """ 
        Creates numpy arrays containing a numerical encoding of the data.
        Uncommon words are mapped to <UNK> in order to reduce the dimensionality.
        Considering we have 3 word features and 3 postag features, the dimensionality 
        will be:
        
           3 * (number of unique words + 1)  /* the +1 is for <UNK> */
         + 3 * (number of unique postags)

        """
        number_of_words = self.number_of_words_above_threshold() + 1  # the "+1" is for <UNK>
        number_of_postags = len(self.p2i)
        dim = 3 * number_of_words + 3 * number_of_postags
        print("number of words = ", number_of_words)
        print("number of postags = ", number_of_postags)
        print("dim = ", dim)
        x = np.zeros((len(self.datapoints), dim + 1))  # the +1 is for the bias term
        y = []
        for i in range(len(self.datapoints)):
            w1, t1, w2, t2, w3, t3, action = self.datapoints[i]
            # First change uncommon words to <UNK>
            if self.wordcount[w1] <= self.THRESHOLD:
                w1 = "<UNK>"
            if self.wordcount[w2] <= self.THRESHOLD:
                w2 = "<UNK>"
            if self.wordcount[w3] <= self.THRESHOLD:
                w3 = "<UNK>"
            index = [0]
            index.append(1 + self.word2int(w1))
            index.append(1 + self.word2int(w2) + number_of_words)
            index.append(1 + self.word2int(w3) + number_of_words * 2)
            index.append(1 + self.postag2int(t1) + number_of_words * 3)
            index.append(1 + self.postag2int(t2) + number_of_words * 3 + number_of_postags)
            index.append(1 + self.postag2int(t3) + number_of_words * 3 + number_of_postags * 2)
            for j in index:
                x[i][j] = 1.
            y.append(action)
        y = np.array(y)
        return x, y

