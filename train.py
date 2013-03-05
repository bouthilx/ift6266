import theano
import pylearn2
from pylearn2.config import yaml_parse
from pylearn2.datasets import preprocessing
import contest_dataset
import sys, getopt



try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:",["ifile="])
except getopt.GetoptError:
    print 'train.py -i <inputfile>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
train = open(inputfile, 'r').read()
train = yaml_parse.load(train)
train.main_loop()
