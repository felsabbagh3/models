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
                if "_LAYER_" in v.tag:
                    x = True
                    value = abs(np.frombuffer(v.tensor.tensor_content, dtype=np.float32)[0])
                    if v.tag in tensors_grad.keys():
                        tensors_grad[v.tag]  += value
                        tensors_count[v.tag] += 1
                    else:
                        tensors_grad[v.tag]  = value
                        tensors_count[v.tag] = 1

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

    sorted_filters = []

    sorted_x = sorted(tensors_grad.items(), key=operator.itemgetter(1))
    for filter_name, filter_grad in sorted_x[::-1]:
        filter_avg = filter_grad / tensors_count[filter_name]
        sorted_filters.append((filter_name, filter_avg))
        # print("{}   :  {}".format(filter_name, filter_avg))

    with gzip.GzipFile(savepath, 'wb') as fid:
        fid.write(pickle.dumps(sorted_filters))

    numPrune = int(len(sorted_filters) * 0.1)
    prune_list = {}
    i = 0
    for rank, value in enumerate(sorted_filters):
        unprocessed_name, avg = value
        unprocessed_name = unprocessed_name.replace("VarName", "")
        unprocessed_name = unprocessed_name.replace("__0", ":0")
        processed_name, layer = unprocessed_name.split("_LAYER_")

        if processed_name in prune_list.keys():
            prune_list[processed_name].append(int(layer))
        else:
            prune_list[processed_name] = [int(layer)]

        # print "{}: {} AND LAYER {}".format(rank, processed_name, layer)
        i += 1
        if (i > numPrune):
            break

    pprint(prune_list)
    with gzip.GzipFile(savepath_pd, 'wb') as fid:
        fid.write(pickle.dumps(prune_list))