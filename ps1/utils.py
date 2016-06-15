import numpy as np

def data_reader(prefix):
    ''' For reading the regression data '''
    parser = lambda line: np.fromiter(map(float, line.strip().split()), dtype=float)

    xfile = open(prefix + "_x.txt")
    X = np.array(list(map(parser, xfile.readlines())))
    yfile = open(prefix + "_y.txt")
    y = np.fromiter(map(float, yfile.readlines()), dtype=int)
    assert len(X) == len(y)
    return X, y
