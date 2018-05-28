import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import gzip


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Dump Gradients from Tensorflow Summaries')
    parser.add_argument('--sum_file', dest='sum_file',
                        help='Which Summary File to Load',
                        default='mytrain/ssd_mobilenetv2_reducedcoco/train/events.out.tfevents.1527361362.greenhpc',
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

if __name__ == '__main__':
    args = parse_args()
    print 'Called with Args'
    print args

    path = args.sum_file
    savepath = os.path.dirname(path) + '/gradients.gz'

    grad_dict = {}

    summary_iterator = ExceptionHandlingIterator(tf.train.summary_iterator(path))
    for summary in summary_iterator:
        for v in summary.summary.value:
            if v.tensor:
                if "VarName" in v.tag:  # == 'gradients/FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/batchnorm/mul_grad/tuple/control_dependency_1VarNameFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/gamma__0':
                    # Get The Tensor Names
                    tag = v.tag.replace("__0", ":0")
                    gradient_tensor_name, variable_tensor_name = tag.split("VarName")

                    # Get the Gradient for that particular tensor
                    grad = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                    shape = []
                    for d in v.tensor.tensor_shape.dim:
                        shape.append(d.size)
                    grad = grad.reshape(shape)

                    if grad_dict.get(gradient_tensor_name, None) is None:
                        grad_dict[gradient_tensor_name] = {
                            'varname': variable_tensor_name,
                            'count': 1,
                            'sum_grad': grad
                        }
                    else:
                        assert grad_dict[gradient_tensor_name]['sum_grad'].shape == grad.shape

                        grad_dict[gradient_tensor_name]['count'] += 1
                        grad_dict[gradient_tensor_name]['sum_grad'] = grad_dict[gradient_tensor_name]['sum_grad'] + grad

    # Save the file with all the gradients

    with gzip.GzipFile(savepath, 'wb') as fid:
        fid.write(pickle.dumps(grad_dict))
