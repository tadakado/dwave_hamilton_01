# dwave_hamilton_01

Hamilton cycle/path solver with D-Wave ocean SDK.
Hamilton graph data set was taken from https://sites.flinders.edu.au/flinders-hamiltonian-cycle-project/graph-database/.
There are two algorithms:
 (1) hamilton_qubo.py, TSP like algorithm, in which each variable represents city and visit. (NxN variables)
 (2) hamilton_qubo2.py, selection of outgoing and incomming arrows. (depends on the number of edges, up to NxN)
