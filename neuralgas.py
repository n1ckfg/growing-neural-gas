# https://github.com/rendchevi/growing-neural-gas
'''
This is an implementation of Growing Neural Gas (GNG) algorithm, an unsupervised
machine learning model based on Self-organizing Map (SOM) useful for learning
topology of a data.

This method commonly used for data clustering, but there has been some
interesting research to use GNG for performing 3D reconstruction task from raw
point cloud data such as acquired from a 3D scanner or LIDAR. You can learn
more in here.
https://ieeexplore.ieee.org/document/6889546

The GNG module I wrote in this repo is part of my bachelor thesis which
concerned on 3D reconstruction for 3D scanned medical data, you can check out
some of my results below (3D full head reconstruction). I don't have any plan
soon to improve or maintain this GNG module, feel free to tinker with the
module, some major improvement is to integrate 3D reconstruction code within
the module.
'''

from tqdm import tqdm
from sys import stdout

import igraph as ig

import numpy as np
from numpy.linalg import norm

class GrowingNeuralGas:

    def __init__(self, dataset, max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200):
        '''
        ---------------------------------------------
        Growing Neural Gas' Parameter Declarations
        ---------------------------------------------
        1. dataset
        2. max_neurons: Maximum # of neurons generated by the network
        3. max_iter: Maximum # of iterations 
        4. max_age: Maximum # of age 
        5. eb: fraction of distance betweeen nearest neuron and input signal
        6. en: fraction of distance betweeen neighboring neurons and input signal
        7. alpha: multiplying scalar for local error
        8. beta: multiplying scalar for global error
        9. l: learning position iterations
        '''
        # Parameters declared by user
        self.dataset_original = dataset.copy()
        self.dataset = dataset.copy()
        self.max_neurons = max_neurons
        self.max_iter = max_iter
        self.max_age = max_age
        self.eb = eb
        self.en = en
        self.alpha = alpha
        self.beta = beta
        self.l = l
        
        # Variable for tracking learning evolution
        self.verts_evolve = []
        self.edges_evolve = []
        
    def initialize_gng(self):
        '''
        Initialize Growing Neural Gas
        '''
        # Get random datapoints from target dataset
        t0 = np.random.randint(0, int(self.dataset.shape[0] / 2))
        t1 = np.random.randint(int(self.dataset.shape[0]/2), self.dataset.shape[0])
        
        # Initialize Growing Neural Gas 
        self.gng = ig.Graph()
        self.gng.add_vertex(weight = self.dataset[t0,:], error = 0)
        self.gng.add_vertex(weight = self.dataset[t1,:], error = 0)
        self.gng.add_edge(0, 1, age = 0)
        
    def learning_position(self):
        for _ in range(0, self.l):
            # Step 1. Get a random datapoint from target dataset
            t = np.random.randint(0, self.dataset.shape[0])
            random_input = self.dataset[t, :]

            # Step 2. Find 2 nearest neuron from random_input
            nearest_index =  np.array([norm(weight - random_input)**2 for weight in self.gng.vs['weight']]).argsort()
            neuron_s1 = self.gng.vs[nearest_index[0]]
            neuron_s2 = self.gng.vs[nearest_index[1]]     

            # Step 3. Increase the age of all neighboring edges from nearest neuron (neuron_s1)
            for edge_id in self.gng.incident(neuron_s1.index):
                self.gng.es[edge_id]['age'] += 1

            # Step 4. Add error to the nearest neuron 
            self.gng.vs[neuron_s1.index]['error'] += norm(neuron_s1['weight'] - random_input)

            # Step 5.1. Update position of nearest neuron
            neuron_s1['weight'] += (self.eb * (random_input - neuron_s1['weight']))
            # Step 5.2. Update position of nearest neuron's neighbors
            for neuron in self.gng.vs[self.gng.neighbors(neuron_s1.index)]:
                neuron['weight'] += (self.en * (random_input - neuron_s2['weight']))

            # Step 6. Update edge of nearest neurons
            EDGE_FLAG = self.gng.get_eid(neuron_s1.index, neuron_s2.index, directed = False, error = False)
            if EDGE_FLAG == -1: # FLAG for no edge detected
                self.gng.add_edge(neuron_s1.index, neuron_s2.index, age = 0)
            else:
                self.gng.es[EDGE_FLAG]['age'] = 0

            # Step 7.1. Delete aging edge 
            for edge in self.gng.es:
                src = edge.source
                tgt = edge.target
                if edge['age'] > self.max_age:
                    self.gng.delete_edges(edge.index)
            # Step 7.2. Delete isolated neuron
            for neuron in self.gng.vs:
                if len(self.gng.incident(neuron)) == 0:
                    self.gng.delete_vertices(neuron)

            # Step 8. Reduce global error
            for neuron in self.gng.vs:
                neuron['error'] *= self.beta

            # Step 9.1. Remove generated random input from target dataset
            self.dataset = np.delete(self.dataset, t, axis = 0)
            # Step 9.2. Reset dataset if datapoints are depleted 
            if self.dataset.shape[0] == 1:
                self.dataset = self.dataset_original.copy()
        
    def update_neuron(self):
        # Adding new neuron from previous learning
        if len(self.gng.vs) <= self.max_neurons:
            # Get neuron q and f
            error_index = np.array([error for error in self.gng.vs['error']]).argsort()
            neuron_q = self.gng.vs[error_index[-1]]
            error = np.array([(neuron['error'], neuron.index) for neuron in self.gng.vs[self.gng.neighbors(neuron_q.index)]])
            error = np.sort(error, axis = 0)
            neuron_f = self.gng.vs[int(error[-1, 1])]
            
            # Add neuron between neuron q and f
            self.gng.add_vertex(weight = (neuron_q['weight'] + neuron_f['weight']) / 2, error = 0)
            neuron_r = self.gng.vs[len(self.gng.vs) - 1]
            
            # Delete edge between neuron q and f
            self.gng.delete_edges(self.gng.get_eid(neuron_q.index, neuron_f.index))
            
            # Create edge between q-r and r-f
            self.gng.add_edge(neuron_q.index, neuron_r.index, age = 0)
            self.gng.add_edge(neuron_r.index, neuron_f.index, age = 0)
            
            # Update neuron error
            neuron_q['error'] *= self.alpha 
            neuron_f['error'] *= self.alpha
            neuron_r['error'] = neuron_q['error']
            
    def learn(self):
        # Initialize GNG
        self.initialize_gng()
        # GNG learning iteration
        for iter, _ in zip(range(0, self.max_iter), tqdm(range(self.max_iter))):
            # Track evolution
            self.verts_evolve.append(np.array([neuron['weight'] for neuron in self.gng.vs]))
            self.edges_evolve.append(np.array([(neuron.source + 1, neuron.target + 1) for neuron in self.gng.es]))
            # Learn new posititon
            self.learning_position()
            self.update_neuron()
            
        return self.gng
