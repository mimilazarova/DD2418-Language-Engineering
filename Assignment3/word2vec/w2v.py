import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__V = 0
        self.__lws = window_size
        self.__rws = window_size
        self.__C = 2*window_size
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling
        self.__W = None
        self.__w2i = {}
        self.__i2w = []
        self.unigram_count = []
        self.unigram_prob = []
        self.corrected_count = []
        self.corrected_prob = []

    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        res = []
        word = ""
        for i in line:
            if i in string.punctuation or i.isdigit():
                continue
            if i in [" ", "\n"]:
                if len(word) > 0:
                    res.append(word)
                    word = ""
            else:
                word = word + i

        if len(word) > 0:
            res.append(word)
        return res

    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        res = []
        l = len(sent)

        # res.extend(sent[max(0, i - self.__lws):i])
        # res.extend(sent[(i + 1):min(l, i + self.__rws + 1)])

        for u in sent[max(0, i - self.__lws):i]:
            res.append(self.__w2i[u])

        for u in sent[(i + 1):min(l, i + self.__rws + 1)]:
            res.append(self.__w2i[u])

        return res

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        #
        # REPLACE WITH YOUR CODE
        #
        focus_words = []
        context_words = []
        for line in self.text_gen():
            for i, v in enumerate(line):

                if v not in self.__i2w:
                    self.__i2w.append(v)
                    ix = self.__i2w.index(v)
                    self.__w2i[v] = ix
                    self.unigram_count.append(1)
                else:
                    ix = self.__i2w.index(v)
                    self.unigram_count[ix] = self.unigram_count[ix] + 1

            for i, v in enumerate(line):
                focus_words.append(self.__w2i[v])
                context_words.append(self.get_context(line, i))

        self.unigram_count = np.array(self.unigram_count)
        self.unigram_prob = self.unigram_count/np.sum(self.unigram_count)
        self.corrected_count = np.power(self.unigram_prob, 0.75)
        self.corrected_prob = self.corrected_count / np.sum(self.corrected_count)
        self.__V = len(self.__i2w)

        return focus_words, context_words

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        list
        """
        #
        # REPLACE WITH YOUR CODE
        #
        if self.__use_corrected:
            counts = self.corrected_count.copy()
        else:
            counts = self.unigram_count.copy()

        counts[xb] = 0.0
        for i in pos:
            counts[i] = 0.0

        dist = counts/np.sum(counts)
        res = np.random.choice(range(self.vocab_size), p=dist, size=number, replace=False)

        return res

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.__W = np.random.random((self.vocab_size, self.__H))
        self.__U = np.random.random((self.vocab_size, self.__H))

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                #
                # YOUR CODE HERE 
                #
                f_ix = x[i]
                pos_ixs = t[i]
                neg_ixs = self.negative_sampling(self.__nsample, f_ix, pos_ixs)
                v = self.__W[f_ix, :]
                v = v.reshape((1, self.__H))
                U_pos = self.__U[pos_ixs, :]
                U_neg = self.__U[neg_ixs, :]

                grad_v = np.dot(U_pos.T, self.sigmoid(np.dot(U_pos, v.T))-1) + \
                         np.dot(U_neg.T, self.sigmoid(np.dot(U_neg, v.T)))

                l_pos = len(pos_ixs)
                vs = np.repeat(v, l_pos, axis=0)
                grad_u_pos = np.multiply(vs, self.sigmoid(np.dot(U_pos, v.T))-1)

                vs = np.repeat(v, self.__nsample, axis=0)
                grad_u_neg = np.multiply(vs, self.sigmoid(np.dot(U_neg, v.T)))

                if self.__use_lr_scheduling:
                    Nt = ep*N + i + 1
                    self.update_lr(Nt, N)

                self.__W[f_ix, :] = v - self.__lr*grad_v.T
                self.__U[pos_ixs, :] = U_pos - self.__lr * grad_u_pos
                self.__U[neg_ixs, :] = U_neg - self.__lr * grad_u_neg

    def update_lr(self, Nt, N):
        lr = self.__lr
        init_lr = self.__init_lr
        if lr < init_lr*0.0001:
            self.__lr = init_lr*0.0001
        else:
            self.__lr = init_lr*(1 - Nt/(self.__epochs*N + 1))

    def get_weights(self):
        return self.__W

    def get_all_words(self):
        return self.__i2w

    def get_word_vector(self, word):
        ix = self.__w2i[word]
        return self.__W[ix, :]

    def find_nearest(self, words, metric, k=5):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        knn = NearestNeighbors(metric=metric)
        knn.fit(self.__W)
        res = []
        for w in words:
            if w in self.__i2w:
                ix = self.__w2i[w]
                vec = self.__W[ix, :]
                neighbours = knn.kneighbors(vec.reshape(1, -1), k)
                ws = [self.__i2w[i] for i in neighbours[1][0]]
                res.append(list(zip(ws, list(np.round(neighbours[0][0], 2)))))
            else:
                res.append([(w, 0.0)])
        return res

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()

# text --> Harry Gryffindor chair wand good enter on school

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors', type=int)
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size', type=int)
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples', type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs', type=int)
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
