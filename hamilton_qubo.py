from collections import defaultdict

def hamilton_qubo(G, cycle=True, lagrange=None):

    # this code is modified from traveling_salesperson_qubo in dwave_networkx

    N = G.number_of_nodes()

    if lagrange is None:
        lagrange = 2

    # Creating the QUBO
    Q = defaultdict(float)

    # Constraint that each row has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += lagrange
                Q[((node, pos_2), (node, pos_1))] += lagrange
                None

    # Constraint that each col has exactly one 1
    for pos in range(N):
        for node_1 in G:
            Q[((node_1, pos), (node_1, pos))] -= lagrange
            for node_2 in set(G)-{node_1}:
                # QUBO coefficient is 2*lagrange, but we are placing this value 
                # above *and* below the diagonal, so we put half in each position.
                Q[((node_1, pos), (node_2, pos))] += lagrange
                None

    # Objective that minimizes distance
    if cycle:
        NN = N
    else:
        NN = N - 1
    for u, v in G.edges:
        for pos in range(NN):
            nextpos = (pos + 1) % N
            Q[((u, pos), (v, nextpos))] -= 1

    return Q


import networkx as nx

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#import itertools

import dimod
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

cycle = True
#cycle = False

in_file = '18_H.txt'
out_file = '18_H.png'

G = nx.DiGraph()

#G.add_edges_from(((0, 1), (1, 0), (0, 2), (0, 3), (1, 2), (3, 1)))

import csv

with open(in_file) as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    for row in reader:
        if len(row) != 2:
            break
        u, v = int(row[0]), int(row[1])
        G.add_edge(u, v)

N = G.number_of_nodes()
if cycle:
    NN = N
else:
    NN = N - 1

Q = hamilton_qubo(G, cycle, 2)

bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

#sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True, 'postprocess': 'sampling'}))
#sampler = SimulatedAnnealingSampler()
#sampler = TabuSampler()
sampler = LeapHybridSampler()

sampleset = sampler.sample(bqm)

route = [None] * N
for (u, t), d in sampleset.first.sample.items():
    if d == 1:
        route[t] = u
        print(u, t)
print(route)

GS = nx.DiGraph()
for i in range(NN):
    ii = (i + 1) % N
    if route[i] is not None and route[ii] is not None:
        GS.add_edge(route[i], route[ii])

pos = nx.spring_layout(G)

plt.figure()
nx.draw_networkx(G,  pos=pos, with_labels=True)
nx.draw_networkx(GS, pos=pos, with_labels=True, edge_color='r')
plt.savefig(out_file, bbox_inches='tight')
