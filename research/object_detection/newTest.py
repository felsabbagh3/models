import sys
from pprint import pprint
import argparse
import cPickle as pickle
import gzip


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Read Filtered List from Tensorflow Summaries')
    parser.add_argument('--filterDir', dest='filterDir',
                        help='Directory containing filter',
                        default='mytrain/ssd_mobilenetv2_reducedcoco/sortedFilters.gz',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args_local = parser.parse_args()
    return args_local


if __name__ == '__main__':
    args = parse_args()
    dire = args.filterDir
    final_dire = dire.replace("sortedFilters", "pruneList")
    with gzip.GzipFile(dire, 'r') as fid:
        sorted_filters = pickle.loads(fid.read())

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