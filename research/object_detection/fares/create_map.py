import cPickle as pickle
from graph_parser import node
from pprint import pprint
import gzip


def get_map(grad_fn, graph_fn):
	grad     = pickle.load(open(grad_fn, 'rb'))
	keys     = grad.keys()
	nodes    = pickle.load(open(graph_fn, 'rb'))
	grad_activation_map = {}
	for curr_grad_name in grad:
		curr_varname = grad[curr_grad_name]['varname'] #Getting the varname
		if (curr_grad_name not in grad_activation_map.keys()): #initializing the activation in dict
			grad_activation_map[curr_grad_name] = ""
		if (":" in curr_varname):
			curr_varname = curr_varname.split(":")[0]
		curr_varname += "/read"                        #Only log read variable operations
		# Looping through all graph nodes
		#i = 0
        	for curr_node in nodes:
			if (curr_node.is_input(curr_varname)):
				if (curr_node.isValid()):
		#			i += 1
		#			if (i > 1):
		#				print("We got a problem: {}".format(curr_varname))
					grad_activation_map[curr_grad_name] += curr_node.getName()
		#if (i == 0):
		#	print("Could not find activation for the following: {}".format(curr_varname))
	return grad_activation_map
if __name__ == '__main__':
	grad_activation_map = get_map("gradients", "graph_def_out.txt")
	#i = 105
	#k = grad_activation_map.keys()[i]
	#v = grad_activation_map[k]
	print("k:{}\nv:{}".format(k,v))
	with gzip.GzipFile('grad_activation_map.gz', 'wb') as fid:
		fid.write(pickle.dumps(grad_activation_map))
