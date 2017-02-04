import os
import struct
import numpy as np

def load_data(path):
    train_img = os.path.join(path, 'train-images-idx3-ubyte')
    train_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    test_img = os.path.join(path, 't10k-images-idx3-ubyte')
    test_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    with open(train_lbl, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_train = np.fromfile(f, dtype=np.int8)

    with open(test_lbl, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_test = np.fromfile(f, dtype=np.int8)
    
    with open(train_img, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X_train = np.fromfile(f, dtype=np.uint8).reshape(len(y_train), rows, cols)
    
    with open(test_img, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X_test = np.fromfile(f, dtype=np.uint8).reshape(len(y_test), rows, cols)

    return (X_train, y_train, X_test, y_test)
