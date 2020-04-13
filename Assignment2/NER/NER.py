import argparse
import sys
import codecs
from BinaryLogisticRegression import BinaryLogisticRegression

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""


class NER(object):
    """
    This class performs Named Entity Recognition (NER).

    It either builds a binary NER model (which distinguishes
    between 'name' or 'noname') from training data, or tries a NER model
    on test data, or both.

    Each line in the data files is supposed to have 2 fields:
    Token, Label

    The 'label' is 'O' if the token is not a name.
    """

    class Dataset(object):
        """
        Internal class for representing a dataset.
        """

        def __init__(self):
            #  The list of datapoints. Each datapoint is itself
            #  a list of features (each feature coded as a number).
            self.x = []

            #  The list of labels for each datapoint. The datapoints should
            #  have the same order as in the 'x' list above.
            self.y = []

    # --------------------------------------------------

    """
    Boolean feature computation. If you want to add more features, add a new
    FeatureFunction class in the 'self.features' array in the __init__ method
    below.
    
    It is often helpful to first write a (boolean) helper method, called from the
    'evaluate' method, as the methods 'capitalizedToken' and 'firstTokenInSentence'
    below.
    """

    def capitalized_token(self):
        return self.current_token != None and self.current_token.istitle()

    def previous_token_capitalized(self):
        return self.last_token != None and self.last_token.istitle()

    def first_token_in_sentence(self):
        return self.last_token in [None, '.', '!', '?']

    class FeatureFunction(object):
        def __init__(self, func):
            self.func = func

        def evaluate(self):
            return 1 if self.func() else 0

    # --------------------------------------------------

    def label_number(self, s):
        return 0 if 'O' == s else 1

    def read_and_process_data(self, filename):
        """
        Read the input file and return the dataset.
        """
        dataset = NER.Dataset()
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f.readlines():
                field = line.strip().split(',')
                if len(field) == 3:
                    # Special case: The token is a comma ","
                    self.process_data(dataset, ',', 'O')
                else:
                    self.process_data(dataset, field[0], field[1])
            return dataset
        return None

    def process_data(self, dataset, token, label):
        """
        Processes one line (= one datapoint) in the input file.
        """
        self.last_token = self.current_token
        self.current_token = token

        datapoint = []
        for f in self.features:
            datapoint.append(f.evaluate())

        dataset.x.append(datapoint)
        dataset.y.append(self.label_number(label))

    def read_model(self, filename):
        """
        Read a model from file
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            d = map(float, f.read().splot(' '))
            return d
        return None

    # ----------------------------------------------------------

    def __init__(self, training_file, test_file, model_file, stochastic_gradient_descent,
                 minibatch_gradient_descent):
        """
        Constructor. Trains and tests a NER model using binary logistic regression.
        """

        self.current_token = None  # The token currently under consideration.
        self.last_token = None  # The token on the preceding line.

        # Here you can add your own features.
        self.features = [
            NER.FeatureFunction(self.capitalized_token),
            NER.FeatureFunction(self.first_token_in_sentence),
            NER.FeatureFunction(self.previous_token_capitalized)
        ]

        if training_file:
            # Train a model
            training_set = self.read_and_process_data(training_file)
            if training_set:
                b = BinaryLogisticRegression(training_set.x, training_set.y)
                if stochastic_gradient_descent:
                    b.stochastic_fit()
                elif minibatch_gradient_descent:
                    b.minibatch_fit()
                else:
                    b.fit()

        else:
            model = self.read_model(model_file)
            if model:
                b = BinaryLogisticRegression(model)

        # Test the model on a test set
        test_set = self.read_and_process_data(test_file)
        if test_set:
            b.classify_datapoints(test_set.x, test_set.y)

    # ----------------------------------------------------------


def main():
    """
    Main method. Decodes command-line arguments, and starts the Named Entity Recognition.
    """

    parser = argparse.ArgumentParser(description='Named Entity Recognition',
                                     usage='\n* If the -d and -t are both given, the program will train a model, and apply it to the test file. \n* If only -t and -m are given, the program will read the model from the model file, and apply it to the test file.')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-t', type=str, required=True, help='test file (mandatory)')

    group = required_named.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', type=str, help='training file (required if -m is not set)')
    group.add_argument('-m', type=str, help='model file (required if -d is not set)')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('-s', action='store_true', default=False, help='Use stochastic gradient descent')
    group2.add_argument('-b', action='store_true', default=False, help='Use batch gradient descent')
    group2.add_argument('-mgd', action='store_true', default=False, help='Use mini-batch gradient descent')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    arguments = parser.parse_args()

    NER(arguments.d, arguments.t, arguments.m, arguments.s, arguments.mgd)

    input("Press Return to finish the program...")


if __name__ == '__main__':
    main()
