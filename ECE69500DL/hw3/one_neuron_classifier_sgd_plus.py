
import sys,os,os.path
import numpy as np
import re
import operator
import math
import random
import torch
from collections import deque
import copy
import matplotlib.pyplot as plt
import networkx as nx
import time

class Exp:
    def __init__(self, exp, body, dependent_var, right_vars, right_params):
        self.exp = exp
        self.body = body
        self.dependent_var = dependent_var
        self.right_vars = right_vars
        self.right_params = right_params

#______________________________  ComputationalGraphPrimer Class Definition  ________________________________

class ComputationalGraphPrimer(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''ComputationalGraphPrimer constructor can only be called with keyword arguments for 
                      the following keywords: expressions, output_vars, dataset_size, grad_delta,
                      learning_rate, display_loss_how_often, one_neuron_model, training_iterations, 
                      batch_size, num_layers, layers_config, epochs, and debug''')
        expressions = output_vars = dataset_size = grad_delta = display_loss_how_often = learning_rate = one_neuron_model = training_iterations = batch_size = num_layers = layers_config = epochs = debug  = None
        if 'one_neuron_model' in kwargs              :   one_neuron_model = kwargs.pop('one_neuron_model')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'num_layers' in kwargs                    :   num_layers = kwargs.pop('num_layers')
        if 'layers_config' in kwargs                 :   layers_config = kwargs.pop('layers_config')
        if 'expressions' in kwargs                   :   expressions = kwargs.pop('expressions')
        if 'output_vars' in kwargs                   :   output_vars = kwargs.pop('output_vars')
        if 'dataset_size' in kwargs                  :   dataset_size = kwargs.pop('dataset_size')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'training_iterations' in kwargs           :   training_iterations = \
                                                                   kwargs.pop('training_iterations')
        if 'grad_delta' in kwargs                    :   grad_delta = kwargs.pop('grad_delta')
        if 'display_loss_how_often' in kwargs        :   display_loss_how_often = kwargs.pop('display_loss_how_often')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug')
        if 'mu' in kwargs                            :   self.mu = kwargs.pop('mu')
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        self.one_neuron_model =  True if one_neuron_model is not None else False
        if training_iterations:
            self.training_iterations = training_iterations
        self.batch_size  =  batch_size if batch_size else 4
        self.num_layers = num_layers 
        if layers_config:
            self.layers_config = layers_config
        if expressions:
            self.expressions = expressions
        if output_vars:
            self.output_vars = output_vars
        if dataset_size:
            self.dataset_size = dataset_size
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if grad_delta:
            self.grad_delta = grad_delta
        else:
            self.grad_delta = 1e-4
        if display_loss_how_often:
            self.display_loss_how_often = display_loss_how_often
        if dataset_size:
            self.dataset_input_samples  = {i : None for i in range(dataset_size)}
            self.true_output_vals       = {i : None for i in range(dataset_size)}
        self.vals_for_learnable_params = None
        self.epochs = epochs
        if debug:                             
            self.debug = debug
        else:
            self.debug = 0
        self.independent_vars = None
        self.gradient_of_loss = None
        self.gradients_for_learnable_params = None
        self.expressions_dict = {}
        self.LOSS = []                               ##  loss values for all iterations of training
        self.all_vars = set()
        if (one_neuron_model is True) or (num_layers is not None):
            self.independent_vars = []
            self.learnable_params = []
        else:
            self.independent_vars = set()
            self.learnable_params = set()
        self.dependent_vars = {}
        self.depends_on = {}                         ##  See Introduction for the meaning of this 
        self.leads_to = {}                           ##  See Introduction for the meaning of this 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_expressions(self):
        ''' 
        This method creates a DAG from a set of expressions that involve variables and learnable
        parameters. The expressions are based on the assumption that a symbolic name that starts
        with the letter 'x' is a variable, with all other symbolic names being learnable parameters.
        The computational graph is represented by two dictionaries, 'depends_on' and 'leads_to'.
        To illustrate the meaning of the dictionaries, something like "depends_on['xz']" would be
        set to a list of all other variables whose outgoing arcs end in the node 'xz'.  So 
        something like "depends_on['xz']" is best read as "node 'xz' depends on ...." where the
        dots stand for the array of nodes that is the value of "depends_on['xz']".  On the other
        hand, the 'leads_to' dictionary has the opposite meaning.  That is, something like
        "leads_to['xz']" is set to the array of nodes at the ends of all the arcs that emanate
        from 'xz'.
        '''
        self.exp_objects = []
        for exp in self.expressions:
            left,right = exp.split('=')
            self.all_vars.add(left)
            self.expressions_dict[left] = right
            self.depends_on[left] = []
            parts = re.findall('([a-zA-Z]+)', right)
            right_vars = []
            right_params = []
            for part in parts:
                if part.startswith('x'):
                    self.all_vars.add(part)
                    self.depends_on[left].append(part)
                    right_vars.append(part)
                else:
                    if self.one_neuron_model is True:
                        self.learnable_params.append(part)
                    else:
                        self.learnable_params.add(part)
                    right_params.append(part)
            exp_obj = Exp(exp, right, left, right_vars, right_params)
            self.exp_objects.append(exp_obj)
        if self.debug:
            print("\n\nall variables: %s" % str(self.all_vars))
            print("\n\nlearnable params: %s" % str(self.learnable_params))
            print("\n\ndependencies: %s" % str(self.depends_on))
            print("\n\nexpressions dict: %s" % str(self.expressions_dict))
        for var in self.all_vars:
            if var not in self.depends_on:              # that is, var is not a key in the depends_on dict
                if self.one_neuron_model is True:
                    self.independent_vars.append(var)                
                else:
                    self.independent_vars.add(var)
        self.input_size = len(self.independent_vars)
        if self.debug:
            print("\n\nindependent vars: %s" % str(self.independent_vars))
        self.dependent_vars = [var for var in self.all_vars if var not in self.independent_vars]
        self.output_size = len(self.dependent_vars)
        self.leads_to = {var : set() for var in self.all_vars}
        for k,v in self.depends_on.items():
            for var in v:
                self.leads_to[var].add(k)    
        if self.debug:
            print("\n\nleads_to dictionary: %s" % str(self.leads_to))


    ### Introduced in 1.0.5
    ######################################################################################################
    ######################################### one neuron model ###########################################
    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        import copy
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)
        self.momentum = copy.deepcopy(self.vals_for_learnable_params)
        for i in self.momentum.keys():
            self.momentum[i] = 0
        self.m_bias = 0
        # self.mu = 0.99

        class DataLoader:
            """
            The data loader's job is to construct a batch of randomly chosen samples from the
            training data.  But, obviously, it must first associate the class labels 0 and 1 with
            the training data supplied to the constructor of the DataLoader.   NOTE:  The training
            data is generated in the Examples script by calling 'cgp.gen_training_data()' in the
            ****Utility Functions*** section of this file.  That function returns two normally
            distributed set of number with different means and variances.  One is for key value '0'
            and the other for the key value '1'.  The constructor of the DataLoader associated a'
            class label with each sample separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):    
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]                
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)

        return loss_running_record

    def forward_prop_one_neuron_model(self, data_tuples_in_batch):
        """
        As the one-neuron model is characterized by a single expression, the main job of this function is
        to evaluate that expression for each data tuple in the incoming batch.  The resulting output is
        fed into the sigmoid activation function and the partial derivative of the sigmoid with respect
        to its input calculated.

        See Slides 103 through 108 of Week 3 slides for the logic implemented  here.
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        """
        output_vals = []
        deriv_sigmoids = []
        for vals_for_input_vars in data_tuples_in_batch:
            input_vars = self.independent_vars                   ## this is just a list of vars for input nodes
            vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
            exp_obj = self.exp_objects[0]
            output_val = self.eval_expression(exp_obj.body , vals_for_input_vars_dict, self.vals_for_learnable_params)
            output_val = output_val + self.bias
            ## apply sigmoid activation (output confined to [0.0,1.0] interval)
            output_val = 1.0 / (1.0 + np.exp(-1.0 * output_val))
            ## calculate partial of the activation function as a function of its input
            deriv_sigmoid = output_val * (1.0 - output_val)
            output_vals.append(output_val)
            deriv_sigmoids.append(deriv_sigmoid)
        return output_vals, deriv_sigmoids


    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slides 103 through 108 of Week 3 slides for the logic implemented  here.
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## calculate the next step in the parameter hyperplane
            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid  \
                   + self.mu * self.momentum[param]
            self.vals_for_learnable_params[param] += step
            self.momentum[param] = step
        step = self.mu * self.m_bias + self.learning_rate * y_error * deriv_sigmoid
        self.bias += step    ## the step to take for the bias
        self.m_bias = step
    ######################################################################################################


    def eval_expression(self, exp, vals_for_vars, vals_for_learnable_params, ind_vars=None):
        self.debug1 = False
        if self.debug1:
            print("\n\nSTEP1: [original expression] exp: %s" % exp)
        if ind_vars is not None:
            for var in ind_vars:
                exp = exp.replace(var, str(vals_for_vars[var]))
        else:
            for var in self.independent_vars:
                exp = exp.replace(var, str(vals_for_vars[var]))
        if self.debug1:
            print("\n\nSTEP2: [replaced ars by their vals] exp: %s" % exp)
        for ele in self.learnable_params:
            exp = exp.replace(ele, str(vals_for_learnable_params[ele]))
        if self.debug1:                     
            print("\n\nSTEP4: [replaced learnable params by their vals] exp: %s" % exp)
        return eval( exp.replace('^', '**') )


    def gen_training_data(self):
        """
        This 2-class dataset is used for the demos in the following Examples directory scripts:

                    one_neuron_classifier.py
                    multi_neuron_classifier.py
                    multi_neuron_classifier.py
 
        The classes are labeled 0 and 1.  All of the data for class 0 is simply a list of 
        numbers associated with the key 0.  Similarly all the data for class 1 is another list of
        numbers associated with the key 1.  

        For each class, the dataset starts out as being standard normal (zero mean and unit variance)
        to which we add a mean value of 2.0 for class 0 and we add mean value of 4 to the square of
        the original numbers for class 1.
        """
        num_input_vars = len(self.independent_vars)
        training_data_class_0 = []
        training_data_class_1 = []
        for i in range(self.dataset_size //2):
            # Standard normal means N(0,1), meaning zero mean and unit variance
            # Such values are significant in the interval [-3.0,+3.0]
            for_class_0 = np.random.standard_normal( num_input_vars )
            for_class_1 = np.random.standard_normal( num_input_vars )
            # translate class_1 data so that the mean is shifted to +4.0 and also
            # change the variance:
            for_class_0 = for_class_0 + 2.0
            for_class_1 = for_class_1 * 2 + 4.0
            training_data_class_0.append( for_class_0 )
            training_data_class_1.append( for_class_1 )
        training_data = {0 : training_data_class_0, 1 : training_data_class_1}
        return training_data

#_________________________  End of ComputationalGraphPrimer Class Definition ___________________________


#______________________________    Test code follows    _________________________________

if __name__ == '__main__':
    import random
    import numpy

    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    # import sys
    # sys.path.append("/home/yangbj/695/hw3/ComputationalGraphPrimer-1.0.8/")
    # from ComputationalGraphPrimer import *

    cgp1 = ComputationalGraphPrimer(
        one_neuron_model=True,
        expressions=['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars=['xw'],
        dataset_size=5000,
        learning_rate=1e-3,
        #               learning_rate = 5 * 1e-2,
        training_iterations=40000,
        batch_size=8,
        display_loss_how_often=100,
        debug=True,
        mu = 0.99
    )

    cgp1.parse_expressions()

    training_data = cgp1.gen_training_data()
    loss_running_record1 = cgp1.run_training_loop_one_neuron_model(training_data)


    cgp2 = ComputationalGraphPrimer(
        one_neuron_model=True,
        expressions=['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars=['xw'],
        dataset_size=5000,
        learning_rate=1e-3,
        #               learning_rate = 5 * 1e-2,
        training_iterations=40000,
        batch_size=8,
        display_loss_how_often=100,
        debug=True,
        mu = 0
    )

    cgp2.parse_expressions()
    # cgp2.display_network2()
    loss_running_record2 = cgp2.run_training_loop_one_neuron_model(training_data)

    plt.figure()
    plt.plot(loss_running_record1)
    plt.plot(loss_running_record2)
    plt.show()
    plt.savefig("/home/yangbj/695/hw3/imgs/" + "one_neuron_loss" + ".jpg")