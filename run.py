# Copyright (c) 2020 Tadashi Kadowaki.
#
# example code for hp_qubo
#

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import dimod
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

from hp_qubo import *

#opts = {'in_file': {'file': 'examples/0012_1.txt'}, 'plot': {'file': 'examples/0012_1.png'}}
#opts = {'in_file': {'file': 'examples/0012_2.txt'}, 'plot': {'file': 'examples/0012_2.png'}}
#opts = {'in_file': {'file': 'examples/0100_1.txt'}, 'out_file': {'results': 'examples/0100_1_out.txt'}}
#opts = {'in_file': {'file': 'examples/0100_2.txt'}, 'out_file': {'results': 'examples/0100_2_out.txt'}}
#opts = {'in_file': {'file': 'examples/1000_1.txt'}, 'out_file': {'results': 'examples/1000_1_out.txt'}}
#opts = {'in_file': {'file': 'examples/1000_2.txt'}, 'out_file': {'results': 'examples/1000_2_out.txt'}}
#opts = {'in_file': {'file': 'examples/0012_1.txt'}, 'plot': {'file': 'examples/0012_1.png'}}
#opts = {'in_file': {'file': 'examples/0012_1.txt'}, 'plot': {'file': 'examples/0012_1.png'}}
#opts = {'in_file': {'file': 'examples/0012_1.txt'}, 'plot': {'file': 'examples/0012_1.png'}}
#opts = {'generate': {'type': 'acyclic', 'n': 12, 'a': 2}, 'out_file': {'problem': 'examples/0012_1.txt'}, 'plot': {'file': 'examples/0012_1.png'}}
#opts = {'generate': {'type': 'acyclic', 'n': 12, 'a': 2}, 'out_file': {'problem': 'examples/0012_2.txt'}, 'plot': {'file': 'examples/0012_2.png'}}
#opts = {'generate': {'type': 'acyclic', 'n': 100, 'a': 2}, 'out_file': {'problem': 'examples/0100_1.txt', 'results': 'examples/0100_1_out.txt'}}
#opts = {'generate': {'type': 'acyclic', 'n': 100, 'a': 5}, 'out_file': {'problem': 'examples/0100_2.txt', 'results': 'examples/0100_2_out.txt'}}
#opts = {'generate': {'type': 'acyclic', 'n': 1000, 'a': 2}, 'out_file': {'problem': 'examples/1000_1.txt', 'results': 'examples/1000_1_out.txt'}}
#opts = {'generate': {'type': 'acyclic', 'n': 1000, 'a': 5}, 'out_file': {'problem': 'examples/1000_2.txt', 'results': 'examples/1000_2_out.txt'}}
opts = {'in_file': {'file': 'examples/DRB1-3123.gfa'}, 'out_file': {'results': 'examples/DRB1-3123_out.txt', 'nodes': 'examples/DRB1-3123_nodes.txt'}}

# Read or generage a graph
if 'in_file' in opts:
    G = read_graph(**opts['in_file'])
elif 'generate' in opts:
    G = generate_graph(**opts['generate'])
else:
    print('Input data is not specified.')
    raise

for e in nx.selfloop_edges(G):
    G.remove_edge(*e)

# Generate QUBO
Q, offset, b, f1 = hamilton_qubo(G, True, False)
bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
bqm.offset = offset

# Fix variables
bqm.fix_variables(f1)
f2 = dimod.fix_variables(bqm)
bqm.fix_variables(f2)
f0 = {**f1, **f2}

print('# of nodes, edges, variables, fixed 1, 2 & total, energy, node with no inedge, multi inedges, no outedges, multi outedges, cycles')
print(G.number_of_nodes(), G.number_of_edges(), bqm.num_variables, len(f1), len(f2), len(f0), end=' ')

# Choose one of the solvers below.
#sampler = SimulatedAnnealingSampler()
sampler = TabuSampler()
#sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True, 'postprocess': 'sampling'}))
#sampler = LeapHybridSampler()

# Conduct optimization
sampleset = sampler.sample(bqm)

print(sampleset.first.energy, end=' ')

# Summarize the results on the graph
GS = sample_graph(G, b, f0, sampleset.first.sample)

# Report violations
rep = report_graph(GS, G)

print(' '.join(str(x) for x in rep), end=' ')

# Report cycles
print(len(list(nx.simple_cycles(GS))))

# output
if 'out_file' in opts:
    if 'problem' in opts['out_file']:
        with open(opts['out_file']['problem'], 'w') as f:
            for u, v in G.edges:
                if isinstance(u, list):
                    u = ''.join(str(x) for x in u)
                if isinstance(v, list):
                    v = ''.join(str(x) for x in v)
                f.write("%s %s\n" % (str(u), str(v)))
    if 'results' in opts['out_file']:
        with open(opts['out_file']['results'], 'w') as f:
            for u, v in sorted(GS.edges):
                if isinstance(u, tuple):
                    u = ''.join(str(x) for x in u)
                if isinstance(v, tuple):
                    v = ''.join(str(x) for x in v)
                f.write("%s %s\n" % (str(u), str(v)))
    if 'nodes' in opts['out_file']:
        GX = GS.copy()
        while True:
            try:
                es = sorted(nx.algorithms.cycles.find_cycle(GX))
                GX.remove_edge(*es[-1])
            except:
                break
        with open(opts['out_file']['nodes'], 'w') as f:
            for n in nx.algorithms.dag.lexicographical_topological_sort(GX):
                if isinstance(n, tuple):
                    n = ''.join(str(x) for x in n)
                f.write("%s\n" % str(n))

# plot
if 'plot' in opts:
    # pos = nx.spring_layout(sorted(G.noeds))
    pos = nx.circular_layout(sorted(G.nodes))
    plt.figure()
    nx.draw_networkx(G,  pos=pos, with_labels=True)
    nx.draw_networkx(GS, pos=pos, with_labels=True, edge_color='r')
    plt.savefig(opts['plot']['file'], bbox_inches='tight')
