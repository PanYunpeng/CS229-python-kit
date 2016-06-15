import pathlib

import numpy as np
from MatrixReader import read_matrix

class NaiveBayesClassifier:
    def __init__(self, trainfile):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

def simple_test(classifier, test_input, labels):
    m = test_input.shape[0]
    correct = sum(1 for i in range(m)
                  if classifier.predict(test_input[i]) == labels[i])
    err = (m - correct) / m
    return err

def test_classifier(classifier, test_input, labels):
    m, n = test_input.shape
    correct_spam, identified_spam, actual_spam = 0, 0, 0
    for i in range(m):
        if classifier.predict(test_input[i]) == 1:
            identified_spam += 1
            if labels[i] == 1:
                correct_spam += 1
        if labels[i] == 1:
            actual_spam += 1

    def trydiv(a,b):
        try:
            return a/b
        except ZeroDivisionError:
            return float('NaN')

    precision = trydiv(correct_spam, identified_spam)
    recall = trydiv(correct_spam, actual_spam)
    f1 = 2 * trydiv(precision * recall , precision + recall)

    print('Precision:  {:.05}'.format(precision))
    print('Recall:     {:.05}'.format(recall))
    print('F-1 score:  {:.05}'.format(f1))

if __name__ == '__main__':
    spam_home = 'data/spam_data/'
    home = pathlib.Path(spam_home)

    NB = NaiveBayesClassifier(home/'MATRIX.TRAIN')
    test_matrix, _, labels = read_matrix((home/'MATRIX.TEST').as_posix())

    test_classifier(NB, test_matrix, labels)
