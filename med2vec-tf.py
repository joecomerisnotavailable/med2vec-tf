"""
# TF code written by Joe Comer (joecomerisnotavailable@gmail.com)
# Adapted from code written by Edward Choi (mp2893@gatech.edu)
# Original implementation in Theano: https://github.com/mp2893/med2vec
"""

import sys
import random
import numpy as np
import pickle
from collections import OrderedDict
import argparse

import tensorflow as tf
from tensorflow.keras import layers

def load_data(xfile, dfile, yfile):
    """Load data."""
    seq_x = np.array(pickle.load(open(xfile, 'rb')))
    if len(dfile) > 0:
        seq_d = np.asarray(pickle.load(open(dfile, 'rb')), dtype=tf.float32)
    else:
        seq_d = []
    if len(yfile) > 0:
        seq_y = np.array(pickle.load(open(yfile, 'rb')))
    else:
        seq_y = []
    return seq_x, seq_d, seq_y

def pick_two(codes, ivector, jvector):
    for first in codes:
        for second in codes:
            if first != second:
                ivector.append(first)
                jvector.append(second)


