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

def hamilton_qubo(G, lagrange=None):
    # This algorithm does not care about path or cycle.

    if lagrange is None:
        lagrange = 2

    Q = defaultdict(float)

    eo, ei = edgelist(G)

    # Constraint that each node has a outgoing arrow.
    for fr, tos in eo.items():
        #print(fr, tos)
        for to in tos:
            Q[((fr, to), (fr, to))] -= lagrange
            for to1, to2 in itertools.combinations(tos, 2):
                Q[((fr, to1), (fr, to2))] += 2 * lagrange

    # Constraint that each node has an incomming arrow.
    for to, frs in ei.items():
        #print(frs, to)
        for fr in frs:
            Q[((fr, to), (fr, to))] -= lagrange
            for fr1, fr2 in itertools.combinations(frs, 2):
                Q[((fr1, to), (fr2, to))] += 2 * lagrange

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

in_file = '18_H.txt'
out_file = '18_H.png'
# Graph data to be used : 0 .. 
idx_data = 2

G = read_bigraph(in_file, idx_data)

Q = hamilton_qubo(G)

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

edges = list(G.edges)
GS = nx.DiGraph()
for (u, v), d in sampleset.first.sample.items():
    if d == 1:
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
