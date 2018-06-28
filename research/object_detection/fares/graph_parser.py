import argparse
from pprint import pprint
import cPickle as pickle


class node:

    def __init__(self, node_text):
        del node_text[0]
        del node_text[-1]
        self.ident = dict()
        for curr_line_index in range(len(node_text)):
            curr_line = node_text[curr_line_index]
            curr_line = curr_line.replace(" ", "")
            curr_line = curr_line.replace('"', "")
            curr_line = curr_line.replace("'", "")
            curr_line = curr_line.replace("\n", "")
            if (":" in curr_line):
                try:
                    curr_key, curr_value = curr_line.split(":", 1)
                except:
                    print(curr_line)
                    exit()
                if (curr_key in self.ident.keys()):
                    self.ident[curr_key].append(curr_value)
                else:
                    self.ident[curr_key] = [curr_value]

    def is_input(self, other_input):
        if ("input" in self.ident.keys()):
            if (other_input in self.ident["input"]):
                return True
        return False

    def getName(self):
        if ("name" in self.ident.keys()):
            return self.ident["name"][0]
        return "N/A"

    def isValid(self):
        if (("name" in self.ident.keys()) and ("L2Loss" not in self.ident["op"][0])):
            first_path = self.ident['name'][0].split("/", 1)[0]
            if (('FeatureExtractor' in first_path) or ("BoxPredictor" in first_path)):
                return True
        return False

    def __str__(self):
        return self.ident.__str__()

    def __repr__(self):
        return self.ident.__str__()


def parse_file(file_name):
    all_nodes = []
    with open(file_name, 'r') as f:
        curr_node = [f.readline()]
        for line in f:
            if ("node" in line):
                all_nodes.append(node(curr_node))
                curr_node = [line]
            else:
                curr_node.append(line)
    return all_nodes


def output_file(file_name, all_nodes):
    with open(file_name, 'w') as f:
        pickle.dump(all_nodes, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gFile", help="Graph Def of the NN session outputed as a .txt", type=str)
    parser.add_argument("oFile", help="Name of output file that includes all nodes of graph")
    args = parser.parse_args()
    all_nodes = parse_file(args.gFile)
    output_file(args.oFile, all_nodes)


if __name__ == '__main__':
    main()
