import csv
import itertools
import networkx as nx
from collections import defaultdict

def read_bigraph(in_file, idx_data):
    G = nx.DiGraph()

    with open(in_file) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for _ in range(idx_data):
            while len(reader.__next__()) == 2:
                pass
        for row in reader:
            if len(row) != 2:
                break
            u, v = int(row[0]), int(row[1])
            G.add_edge(u, v)
    return G

def edgelist(G):
    eo = defaultdict(list)
    ei = defaultdict(list)
    for st, ens in G.adj.items():
        ens = list(ens.keys())
        for en in ens:
            eo[st] += [en]
            ei[en] += [st]
    return eo, ei

def report_graph(GS):
    eo, ei = edgelist(GS)
    for st in GS.nodes:
        nout = len(eo[st])
        if nout == 0:
            print(st, ": no outputs")
        if nout > 1:
            print(st, ": multiple outputs", eo[st])
    for en in GS.nodes:
        nin = len(ei[en])
        if nin == 0:
            print(en, ": no inputs")
        if nin > 1:
            print(pos, ": multiple inputs", ei[en])

def hamilton_qubo(G, cycle=True, lagrange=None):

    # This code is modified from traveling_salesperson_qubo in dwave_networkx

    N = G.number_of_nodes()

    if lagrange is None:
        lagrange = 2

    # Creating the QUBO
    Q = defaultdict(float)

    # Constraint that each row (city) has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += lagrange
                Q[((node, pos_2), (node, pos_1))] += lagrange
                None

    # Constraint that each col (visit) has exactly one 1
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


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import dimod
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

#cycle = True
cycle = False

in_file = '18_H.txt'
out_file = '18_H.png'
# Graph data to be used : 0 .. 
idx_data = 2

G = read_bigraph(in_file, idx_data)

N = G.number_of_nodes()
if cycle:
    NN = N
else:
    NN = N - 1

Q = hamilton_qubo(G, cycle)

bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

# Choose one from below. The top two samplers are software algorithms,
# while the bottoms are hardware / hybrid algorithms.
#sampler = SimulatedAnnealingSampler()
#sampler = TabuSampler()
#sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True, 'postprocess': 'sampling'}))
sampler = LeapHybridSampler()

# Conduct optimization
sampleset = sampler.sample(bqm)

#print(sampleset.first)

route = defaultdict(list)
for (u, t), d in sampleset.first.sample.items():
    if d == 1:
        route[t] += [u]

#print(dict(route))

edges = list(G.edges)
GS = nx.DiGraph()
for pos in range(NN):
    nextpos = (pos + 1) % N
    for u, v in itertools.product(route[pos], route[nextpos]):
        if (u, v) in edges:
            GS.add_edge(u, v)
        else:
            print(u, v, ": not in G")

report_graph(GS)

#pos = nx.spring_layout(G)
pos = nx.circular_layout(G)

plt.figure()
nx.draw_networkx(G,  pos=pos, with_labels=True)
nx.draw_networkx(GS, pos=pos, with_labels=True, edge_color='r')
plt.savefig(out_file, bbox_inches='tight')
