# Copyright (c) 2020 Tadashi Kadowaki.
#
# example code for hp_qubo
#

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import dimod
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

import s_gd2

from hp_qubo import *

def draw(file, G, GS, X):
    colors = ('red', 'green', 'blue', 'orange', 'cyan', 'purple', 'pink', 'sienna', 'seagreen')
    GSS = [GS.subgraph(c) for c in nx.connected_components(GS.to_undirected())]
    GSS = sorted(GSS, key=len, reverse=True)
    v2s = {}
    for i in range(len(GSS)):
        v2s.update(dict(zip(list(GSS[i]), [i] * len(GSS[i]))))
    for v in G:
        if v2s.get(v, None) is None:
            v2s[v] = -1
    C = []
    W = []
    O = []
    for u, v in G.edges:
        c = 'black'
        w = 1
        o = 1
        for i in range(len(GSS)):
            if (u, v) in GSS[i].edges:
                c = colors[i % len(colors)]
        if c == 'black' and (v2s[u] != v2s[v] or v2s[u] == -1 or v2s[v] == -1):
            w = 3
            o = 0.3
        C += [c]
        W += [w]
        O += [o]
    draw_svg(X, I, J, C, W, O, file, width=5000, noderadius=0.1, linkwidth=0.2)

def draw_svg(X, I, J, C, W, O, filepath=None, noderadius=.2, linkwidth=.05, width=1000, border=50, nodeopacity=1, linkopacity=1):
    """Takes a n-by-2 matrix of positions X and index pairs I and J
    and writes the equivalent picture in svg format.
    The drawing will be expanded into a width*width square
    Note that the parameters are in svg pixel units.
    The style at the top of the output svg may also be edited as necessary.
    The svg is returned as a string if filename is empty."""

    n = len(X)
    m = len(I)

    X_min = [min(X[:,0]), min(X[:,1])]
    X_max = [max(X[:,0]), max(X[:,1])]
    range_max = max(X_max[0]-X_min[0], X_max[1]-X_min[1]) # taller or wider
    range_max += 2*noderadius # guarantee no nodes are cut off at the edges
    scale = (width-2*border) / range_max

    X_svg = np.empty((n,2))
    for i in range(n):
        X_svg[i] = (X[i] - X_min) * scale
        X_svg[i] += [border + scale*noderadius, border + scale*noderadius]

    svg_list = []
    svg_list.append('<svg width="{:.0f}" height="{:.0f}" xmlns="http://www.w3.org/2000/svg">'.format(width, width))
    svg_list.append('<style type="text/css">')
    svg_list.append('line{{stroke:black;stroke-width:{:.3f};stroke-opacity:{:.3f};stroke-linecap:round;}}'.format(scale*linkwidth,linkopacity))
    svg_list.append('circle{{r:{};fill:black;fill-opacity:{:.3f}}}'.format(scale*noderadius,nodeopacity))
    svg_list.append('</style>')

    # draw links
    for ij in range(m):
        i = I[ij]
        j = J[ij]
        X_i = X_svg[i]
        X_j = X_svg[j]
        style = []
        if C[ij] != 'black':
            style += ['stroke:{:s}'.format(C[ij])]
        if W[ij] != 1:
            style += ['stroke-width:{:.3f}'.format(scale*linkwidth*W[ij])]
        if O[ij] != 1:
            style += ['stroke-opacity:{:.3f}'.format(linkopacity*O[ij])]
        style = ';'.join(style)
        if style != '':
            svg_list.append('<line x1="{:.1f}" x2="{:.1f}" y1="{:.1f}" y2="{:.1f}" style="{:s}"/>'.format(X_i[0], X_j[0], X_i[1], X_j[1], style))
        else:
            svg_list.append('<line x1="{:.1f}" x2="{:.1f}" y1="{:.1f}" y2="{:.1f}"/>'.format(X_i[0], X_j[0], X_i[1], X_j[1]))

    # draw nodes
    if noderadius > 0:
        for i in range(n):
            svg_list.append('<circle cx="{:.1f}" cy="{:.1f}"/>'.format(X_svg[i][0], X_svg[i][1]))

    svg_list.append("</svg>")

    if filepath is None:
        return '\n'.join(svg_list)
    else:
        f = open(filepath, 'w')
        f.write('\n'.join(svg_list))
        f.close()

def connected_subgraph(G):
    cc = [c for c in nx.connected_components(G.to_undirected())]
    cc = sorted(cc, key=len, reverse=True)
    return G.subgraph(cc[0])

file = 'examples/DRB1-3123.gfa'

G = read_graph(file)
#G = read_gfa_path(file)

G = connected_subgraph(G)

##########

def solve(Q, offset, b, f1, sampler):
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    bqm.fix_variables(f1)
    f2 = dimod.fix_variables(bqm)
    bqm.fix_variables(f2)
    f = {**f1, **f2}
    sampleset = sampler.sample(bqm)
    GS = sample_graph(G, b, f, sampleset.first.sample)
    return GS, sampleset, sampleset.first.energy, GS.edges

sampler = LeapHybridSampler()
#sampler = SimulatedAnnealingSampler()
#sampler = TabuSampler()

#G = nx.DiGraph([(1,2),(1,3),(1,4),(2,5),(3,5),(4,5),(5,6),(5,7),(5,8),(6,9),(7,9),(8,9)]) 

v2i = dict(list(zip(list(G), list(range(len(G))))))
I = []
J = []
for u, v in G.edges:
    I += [v2i[u]]
    J += [v2i[v]]
X = s_gd2.layout(I, J)

for run in range(5):
    #####

    Q, offset, b, f1 = hamilton_qubo(G, lagrange=2, fix_var=False, reduce_var=False, relax=False, elongation=False, weight=False)
    GS_01, sampleset, energy, edges = solve(Q, offset, b, f1, sampler)
    print("Sampler information (processing time)")
    print(sampleset.info)
    print(len(edges), energy)
    #print(edges)

    #####

    Q, offset, b, f1 = hamilton_qubo(G, lagrange=2, fix_var=False, reduce_var=False, relax=True, elongation=False, weight=False)
    GS_02, sampleset, energy, edges = solve(Q, offset, b, f1, sampler)
    print("Sampler information (processing time)")
    print(sampleset.info)
    print(len(edges), energy)
    #print(edges)

    #####

    Q, offset, b, f1 = hamilton_qubo(G, lagrange=2, fix_var=False, reduce_var=False, relax=False, elongation=True, weight=False)
    GS_03, sampleset, energy, edges = solve(Q, offset, b, f1, sampler)
    print("Sampler information (processing time)")
    print(sampleset.info)
    print(len(edges), energy)
    #print(edges)

    #####

    Q, offset, b, f1 = hamilton_qubo(G, lagrange=2, fix_var=False, reduce_var=False, relax=True, elongation=True, weight=False)
    GS_04, sampleset, energy, edges = solve(Q, offset, b, f1, sampler)
    print("Sampler information (processing time)")
    print(sampleset.info)
    print(len(edges), energy)
    #print(edges)

    #####

    Q, offset, b, f1 = hamilton_qubo(G, lagrange=2, fix_var=False, reduce_var=False, relax=True, elongation=True, weight=True)
    GS_05, sampleset, energy, edges = solve(Q, offset, b, f1, sampler)
    print("Sampler information (processing time)")
    print(sampleset.info)
    print(len(edges), energy)
    #print(edges)

    #####

    print("Size of connected components for different options")
    print(sorted([len(c) for c in nx.connected_components(GS_01.to_undirected())], reverse=True))
    print(sorted([len(c) for c in nx.connected_components(GS_02.to_undirected())], reverse=True))
    print(sorted([len(c) for c in nx.connected_components(GS_03.to_undirected())], reverse=True))
    print(sorted([len(c) for c in nx.connected_components(GS_04.to_undirected())], reverse=True))
    print(sorted([len(c) for c in nx.connected_components(GS_05.to_undirected())], reverse=True))

    print("Sizes of edges and the differences")
    print(len(list(G.edges)), len(list(GS_03.edges)), len(list(GS_04.edges)), len(set(list(GS_03.edges)).symmetric_difference(set(list(GS_04.edges)))))
    print(len(list(G.edges)), len(list(GS_04.edges)), len(list(GS_05.edges)), len(set(list(GS_04.edges)).symmetric_difference(set(list(GS_05.edges)))))

    draw("examples/test01/01_run%02d.svg" % run, G, GS_01, X)
    draw("examples/test01/02_run%02d.svg" % run, G, GS_02, X)
    draw("examples/test01/03_run%02d.svg" % run, G, GS_03, X)
    draw("examples/test01/04_run%02d.svg" % run, G, GS_04, X)
    draw("examples/test01/05_run%02d.svg" % run, G, GS_05, X)
