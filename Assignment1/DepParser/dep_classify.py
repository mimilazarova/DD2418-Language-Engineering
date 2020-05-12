import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        """
        #
        # YOUR CODE HERE
        #
        i, stack, pred_tree = 0, [], [0]*len(words)
        i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, 0)

        while i < len(words) or len(stack) > 1:
            x = ds.dp2array(words, tags, i, stack)
            x = np.concatenate((np.ones(1), x))

            probs = [0, 0, 0]
            for label in range(3):
                p = model.conditional_prob(label, x)
                probs[label] = p

            valid = self.__parser.valid_moves(i, stack, pred_tree)
            m = probs.index(max(probs))
            if m not in valid:
                probs[m] = -1
                m = probs.index(max(probs))
                if m not in valid:
                    probs[m] = -1
                    m = probs.index(max(probs))

            i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, m)

        return pred_tree

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #

        uas_numer = 0
        uas_denom = 0

        sen_numer = 0
        sen_denom = 0

        with open(test_file, encoding='utf-8') as source:
            for w, tags, tree, relations in p.trees(source):
                pred_tree = self.build(model, w, tags, ds)

                sen_denom = sen_denom + 1
                uas_denom = uas_denom + len(w)

                sen_numer = sen_numer + (np.array(tree) == np.array(pred_tree)).all()
                uas_numer = uas_numer + np.sum(np.array(tree) == np.array(pred_tree))

        print("UAS accuracy: {0:.2f}%".format(uas_numer*100/uas_denom))
        print("Sentence accuracy: {0:.2f}%".format(sen_numer*100/sen_denom))


if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)
    model_file = 'model.pkl'
    # model_file = 'model_t800.pkl'
    # Train LR model

    if os.path.exists(model_file):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open(model_file, 'rb'))
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open(model_file, 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())
    
    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev.conllu', ds)