# Copyright (c) 2020 Tadashi Kadowaki.
#
# Solve Hamiltonian path by annealing
#

import csv
from itertools import combinations
import networkx as nx
from collections import defaultdict
from random import choices, randint

def generate_graph(type, **kwargs):
    if type == 'random':
        return generate_random_graph(**kwargs)
    elif type == 'acyclic':
        return generate_acyclic_graph(**kwargs)

def generate_random_graph(n=100, a=2):
    G = nx.DiGraph()
    G.add_edges_from(zip(range(1, n), range(2, n+1)))
    G.add_edges_from(zip(choices(range(1, n+1), k=n * a), choices(range(1, n+1), k=n * a)))
    return G

def generate_acyclic_graph(n=100, a=2):
    G = nx.DiGraph()
    G.add_edges_from(zip(range(1, n), range(2, n+1)))
    for i in range(n * a):
        u = randint(1, n-2)
        v = randint(u+1, n)
        G.add_edge(u, v)
    return G

def rev(u, v):
    return (v[0], '-' if v[1] == '+' else '+'), (u[0], '-' if u[1] == '+' else '+')

def simplify_gfa_graph(G):
    GS = nx.DiGraph()
    for u, v in G.edges:
        if (u[1] == '-' and v[1] == '-'):
            u, v = rev(u, v)
        GS.add_edge(u, v)
    return GS

def read_graph(file, **kwargs):
    if file[-4:].upper() == '.GFA':
        G = read_gfa_link(file, **kwargs)
        return simplify_gfa_graph(G)
    elif file[-4:].upper() == '.TXT':
        return read_txt(file, **kwargs)
    # elif ...:
    #     return read_gfa_path(file, **kwargs)

def read_gfa_link(file, verbose=False):
    G = nx.DiGraph()
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0] != 'L':
                continue
            u = (int(row[1]), row[2])
            v = (int(row[3]), row[4])
            if u == v:
                if verbose:
                    print("ignore self loop:", u, v)
                continue
            if G.has_edge(u, v):
                if verbose:
                    print("ignore duplicated edge:", u, v)
                continue
            G.add_edge(u, v)
    G = simplify_gfa_graph(G)
    return G

def read_gfa_path(file, verbose=False, shrink=True):
    GG = []
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0] != 'P':
                continue
            gi = row[1]
            path = row[2]
            G = nx.DiGraph()
            u = None
            for v in path.split(','):
                v = (int(v[:-1]), v[-1])
                if u:
                    if u[0] == v[0]:
                        if verbose:
                            print('selfloop', u, v)
                    else:
                            G.add_edge(u, v)
                u = v
            G = simplify_gfa_graph(G)
            GG += [G]
    if shrink:
        edges = []
        for G in GG:
            edges += list(G.edges)
        GG = nx.DiGraph(sorted(list(set(edges))))
    return GG

def read_txt(file):
    G = nx.DiGraph()
    with open(file) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            if len(row) != 2:
                break
            u, v = int(row[0]), int(row[1])
            G.add_edge(u, v)
    return G

def report_graph(GS, G, verbose=False):
    out = [0, 0, 0, 0]
    for u in GS.nodes:
        n_gs_out = len(set(GS.successors(u)) - {u})
        n_g_out = len(set(G.successors(u)) - {u})
        if n_gs_out == 0 and n_g_out != 0:
            out[0] += 1
            if verbose:
                print(u, ": no outputs")
        if n_gs_out > 1:
            out[1] += 1
            if verbose:
                print(u, ": multiple outputs", list(GS.successors(u)))
    for v in GS.nodes:
        n_gs_in = len(set(GS.predecessors(v)) - {v})
        n_g_in = len(set(G.predecessors(v)) - {v})
        if n_gs_in == 0 and n_g_in != 0:
            out[2] += 1
            if verbose:
                print(v, ": no inputs")
        if n_gs_in > 1:
            out[3] += 1
            if verbose:
                print(v, ": multiple inputs", list(GS.predecessors(v)))
    return out

def hamilton_qubo(G, fix_var=True, reduce_var=False):
    # This algorithm does not care about path or cycle.
    # Inputs
    #   G          : directed graph
    #   fix_var    : fix variables (single edge node)
    #   reduce_var : reduce variables (1 bit encoding for double edge node)
    # Outputs
    #   Q     : QUBO
    #   offset: offset
    #   b     : binary variables
    #   f     : fixed variables
    # [FixMe] reduce_var does not work with fix_var

    Q = defaultdict(float)
    offset = 0
    b = {}
    f = {}

    lagrange = 1

    for u in G:
        vv = list(G.successors(u))
        if len(vv) == 2:
            vv = sorted(vv)
            b[(u, vv[1])] = (u, vv[0])

    # Constraint (outedges)
    for u in G:
        vv = list(G.successors(u))
        if len(vv) == 0:
            continue
        if len(vv) == 1 and fix_var:
            v = vv[0]
            f[(u, v)] = 1
        if len(vv) == 2 and reduce_var:
            continue
        x = [None] * len(vv)
        for i, v in enumerate(vv):
            x[i] = (u, v)
        for i in range(len(x)):
            Q[(x[i], x[i])] += - lagrange
        for i, j in combinations(range(len(x)), 2):
            Q[(x[i], x[j])] += 2 * lagrange
        offset += lagrange

    # Constraint (inedges)
    for v in G:
        uu = list(G.predecessors(v))
        if len(uu) == 0:
            continue
        sign = [1] * len(uu)
        x = [None] * len(uu)
        n = 0
        for i, u in enumerate(uu):
            if ((u, v) in b) and reduce_var:
                sign[i] = -1
                x[i] = b[(u, v)]
                n += 1
            else:
                sign[i] = 1
                x[i] = (u, v)
        if len(uu) == 1 and fix_var:
            if sign[0] == -1:
                f[x[0]] = 0
            else:
                f[x[0]] = 1
        for i in range(len(x)):
            Q[(x[i], x[i])] += (1 + 2*(n-1)*sign[i]) * lagrange
        for i, j in combinations(range(len(x)), 2):
            Q[(x[i], x[j])] += 2 * sign[i] * sign[j] * lagrange
        offset += (n * n - 2 * n + 1) * lagrange

    return Q, offset, b, f

def sample_graph(G, b, f, sample):
    edges = list(G.edges)
    b = dict(zip(b.values(), b.keys()))
    GS = nx.DiGraph()
    for (u, v), d in sample.items():
        if d == 0 and (u, v) in b:
            GS.add_edge(*b[(u, v)])
        if d == 1:
            if (u, v) in edges:
                GS.add_edge(u, v)
            else:
                print(u, v, ": not in G")
    for e, d in f.items():
        if d == 1:
            GS.add_edge(*e)
    return GS

def reverse_graph(G):
    GR = nx.DiGraph()
    for u, v in G.edges:
        GR.add_edge(v, u)
    return GR
