import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import gzip
import glob
import operator
from pprint import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Dump Gradients from Tensorflow Summaries')
    parser.add_argument('--eventDir', dest='eventDir',
                        help='Directory containing event files',
                        default='mytrain/ssd_mobilenetv2_reducedcoco/train/',
                        type=str)
    # parser.add_argument('--feedback', dest='feedback',
    #                     help='whether to apply feedback',
    #                     action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args_local = parser.parse_args()
    return args_local


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


class ExceptionHandlingIterator(object):
    def __init__(self, iterable):
        self._iter = iter(iterable)
        self.handlers = []

    def __iter__(self):
        return self

    def next(self):
        try:
            return self._iter.next()
        except StopIteration as e:
            raise e
        except Exception as e:
            print e
            for handlers in self.handlers:
                pass
            return self.next()


def getGradDict(path, tensors_grad, tensors_count):
    if bool(tensors_grad) is False:
        print "Empty Dictionary"
    else:
        print "Appending to Existing Dictionary"

    # import operator
    # from pprint import pprint
    x = False

    summary_iterator = ExceptionHandlingIterator(tf.train.summary_iterator(path))

    for summary in summary_iterator:
        for v in summary.summary.value:
            if v.tag:
                if "masked/" in v.tag:
                    print v.tag
                    value = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                    print value

    # return grad_dict


if __name__ == '__main__':
    args = parse_args()
    print 'Called with Args'
    print args

    path = args.eventDir
    eventList = sorted(glob.glob(os.path.join(os.path.abspath(path), 'events.out*')))

    tensors_grad  = {}
    tensors_count = {}
    savepath      = os.path.dirname(path) + '/sortedFilters.gz'
    savepath_pd   = os.path.dirname(path) + '/pruneDict.gz'

    for path in eventList:
        print "Fetching gradients from path: {}".format(path)
        getGradDict(path, tensors_grad, tensors_count)

