import pathlib
import random
import math

import numpy as np

from MatrixReader import read_matrix

class SVMClassifier:
    def __init__(self, trainfile):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

def test_svm(classifier, Xtest, ytest):
    num_test = Xtest.shape[0]
    total_correct = 0
    identified_spam, actual_spam, correct_spam = 0, 0, 0

    yhat = np.array([classifier.predict(Xtest[i])
                    for i in range(num_test)])

    yhat_pos = yhat > 0
    ytest_pos = ytest > 0
    identified_spam = np.count_nonzero(yhat_pos)
    actual_spam = np.count_nonzero(ytest_pos)
    correct_spam = np.count_nonzero(yhat_pos & ytest_pos)
    total_correct = np.count_nonzero(yhat == ytest)

    def trydiv(a, b):
        try:
            return a / b
        except DivisionByZeroError:
            return float('NaN')

    precision = trydiv(correct_spam, identified_spam)
    recall = trydiv(correct_spam, actual_spam)
    f1 = 2*trydiv(recall*precision, precision + recall)
    print("Correct identified {}% of result".format(total_correct / num_test * 100))
    print("Precision: {}".format(precision))
    print("Recall:    {}".format(recall))
    print("F-1 score: {}".format(f1))

def simple_test(classifier, Xtest, ytest):
    m = Xtest.shape[0]
    correct = sum(1 for i in range(m)
                  if classifier.predict(Xtest[i]) == ytest[i])
    err = (m - correct) / m
    return err

if __name__ == '__main__':
    from MatrixReader import read_matrix
    spam_home = 'data/spam_data/'
    home = pathlib.Path(spam_home)

    testmat, _, label = read_matrix((home/'MATRIX.TEST').as_posix())
    Xtest = (1*(testmat > 0)).astype(int)   # replace all value > 0 with 1
    ytest = 2*label - 1        # make y values +1 = SPAM, -1 = Non-SPAM

    svm = SVMClassifier((home/'MATRIX.TRAIN.50').as_posix())
    test_svm(svm, Xtest, ytest)
