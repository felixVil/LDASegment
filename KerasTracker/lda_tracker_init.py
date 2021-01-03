from VOTExecutionFuncs import *
import os
from sys import argv
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.setrecursionlimit(10000)
    initialize(*argv[1:])
