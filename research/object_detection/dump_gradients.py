import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import gzip
import glob


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


def getGradDict(path, grad_dict):
    if bool(grad_dict) is False:
        print "Empty Dictionary"
    else:
        print "Appending to Existing Dictionary"

    import operator
    from pprint import pprint
    x = False
    tensors = {}

    summary_iterator = ExceptionHandlingIterator(tf.train.summary_iterator(path))

    for summary in summary_iterator:
        for v in summary.summary.value:
            if v.tag:
                if "_FARES_" in v.tag:
                    x = True
                    value = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                    if v.tag in tensors.keys():
                        tensors[v.tag]['grad']  += value[0]
                        tensors[v.tag]['count'] += 1
                    else:
                        tensors[v.tag]          = {}
                        tensors[v.tag]['grad']  = value[0]
                        tensors[v.tag]['count'] = 1
                    # print "************************************"
            # if v.tensor:
                # if "VarNameRank" in v.tag:
                #     continue
                # elif "VarName" in v.tag:  # == 'gradients/FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/batchnorm/mul_grad/tuple/control_dependency_1VarNameFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/gamma__0':
                #     # Get The Tensor   Names
                #     tag = v.tag.replace("__0", ":0")
                #     _, variable_tensor_name = tag.split("VarName")
                #
                #     # Get the Gradient for that particular tensor
                #     grad = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                #     shape = []
                #     for d in v.tensor.tensor_shape.dim:
                #         shape.append(d.size)
                #     grad = grad.reshape(shape)
                #
                #     if grad_dict.get(variable_tensor_name, None) is None:
                #         grad_dict[variable_tensor_name] = {
                #             'varname': variable_tensor_name,
                #             'count': 1,
                #             'sum_grad': grad
                #         }
                #     else:
                #         assert grad_dict[variable_tensor_name]['sum_grad'].shape == grad.shape
                #
                #         grad_dict[variable_tensor_name]['count'] += 1
                #         grad_dict[variable_tensor_name]['sum_grad'] += grad

    if x:
        # sorted_x = sorted(tensors.items(), key=operator.itemgetter(1))
        pprint(tensors)
        exit()

    return grad_dict


if __name__ == '__main__':
    args = parse_args()
    print 'Called with Args'
    print args

    path = args.eventDir
    eventList = sorted(glob.glob(os.path.join(os.path.abspath(path), 'events.out*')))

    grad_dict = {}
    savepath = os.path.dirname(path) + '/gradients.gz'

    for path in eventList:
        print "Fetching gradients from path: {}".format(path)
        getGradDict(path, grad_dict)

    # Save the file with all the gradients

    # with gzip.GzipFile(savepath, 'wb') as fid:
    #     fid.write(pickle.dumps(grad_dict))
    #
    # dictFile = os.path.dirname(path) + '/gradient_dict.gz'
    # with gzip.GzipFile(dictFile, 'rb') as fid:
    #     infoDict = pickle.loads(fid.read())
    #
    # act_shapes_file = os.path.dirname(path) + '/act_shapes.pkl'
    # with open(act_shapes_file, 'rb') as fid:
    #     act_shapes_dict = pickle.loads(fid.read())
    #
    # # Add info for activation sizes for FLOP computation
    # for varname, vardict in infoDict.iteritems():
    #     grad_dict[varname]['kW'] = int(vardict['kW'])
    #     grad_dict[varname]['kH'] = int(vardict['kH'])
    #     grad_dict[varname]['in_depth'] = int(vardict['in_channels'])
    #     grad_dict[varname]['out_depth'] = int(vardict['out_channels'])
    #     grad_dict[varname]['rank_activation'] = act_shapes_dict[varname]
    #     grad_dict[varname]['FLOPS'] = grad_dict[varname]['kH'] * \
    #                                     grad_dict[varname]['kW'] * \
    #                                         grad_dict[varname]['in_depth'] * \
    #                                             grad_dict[varname]['out_depth'] * \
    #                                                 grad_dict[varname]['rank_activation'][1] * \
    #                                                     grad_dict[varname]['rank_activation'][2]
    #
    #
    # FLOPS = 0
    # for key, value in grad_dict.iteritems():
    #     FLOPS += value['FLOPS']
    #
    # print FLOPS/1e9
    #
    # # Add Code here for pruning
    # print "Making list of every filter in network and their rank"
    # filter_list = []
    # value_list = []
    # for varname, vardict in grad_dict.iteritems():
    #     avg_grad = vardict['sum_grad'] / vardict['count'] / (vardict['rank_activation'][0] * vardict['rank_activation'][1] * vardict['rank_activation'][2])
    #     for filter_i in range(vardict['out_depth']):
    #         filter_list.append([varname, filter_i])
    #         value_list.append(avg_grad[filter_i])
    #
    # # print filter_list
    # #
    # pruning_ratio = 0.1
    # #
    # print "Sorting the Filters according to rank"
    # valueArray = np.array(value_list)
    # print valueArray
    # order = np.argsort(valueArray)
    # count_filters = len(valueArray)
    # #
    # print "Sorted {} Filters".format(count_filters)
    # threshold = valueArray[order[int(count_filters*pruning_ratio)]] # get threshold in sorted array
    # valueMask = (valueArray >= threshold) * 1
    # print threshold
    #
    # from operator import itemgetter
    # sorted_filters = sorted(filter_list,key=itemgetter(order))
    # #
    # print "Kept {} filters with pruning ratio {}".format(np.sum(valueMask),pruning_ratio)