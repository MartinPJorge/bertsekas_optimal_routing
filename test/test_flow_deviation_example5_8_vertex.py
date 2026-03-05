# This tests the flow deviation method in [Example 5.8, 1]
# putting the cost function in the vertex
# [1] Bertsekas, D., & Gallager, R. (2021). Data networks. Athena Scientific.


import networkx as nx
import sys
sys.path.append('..')
from bertsekas_optimal_routing import flow_deviation





# Create the graph in Figure 5.66.
# Numbers represent the nodes
#   1
#  / \
# 0-2-4
#  \ /
#   3
# we create a non-multigraph version to support element hashing
G = nx.Graph()
G.add_edge(0,1)
G.add_edge(1,4)
G.add_edge(0,2)
G.add_edge(2,4)
G.add_edge(0,3)
G.add_edge(3,4)

# Create the list of paths 
od_paths = {0: [[(0,1),(1,4)], [(0,2),(2,4)], [(0,3),(3,4)]]}

# Specify the OD flow requirement
od_req = {0: 1}

# Specify the link costs
# x = (x1,x2,x3)
costs = {
    1: lambda x: x**2/2,
    2: lambda x: x**2/2,
    3: lambda x: x**2/2 + .55*x,
}

# Specify the link cost derivatives
# x = (x1,x2,x3)
costs_ = {
    1: lambda x: x,
    2: lambda x: x,
    3: lambda x: x + .55,
}

# Specify initial solution (all going down)
x0 = {0: [0,0,1]}

# Create the step size function
alpha_fn = lambda xp,x_: -1\
    * (xp[0][0]*(x_[0][0]-xp[0][0])+xp[0][1]*(x_[0][1]-xp[0][1])+(.1*xp[0][2]+.55)*(x_[0][2]-xp[0][2]))\
    / ((x_[0][0]-xp[0][0])**2+(x_[0][1]-xp[0][1])**2+.1*(x_[0][2]-xp[0][2])**2)

flow_deviation(G, od_paths, od_req, costs, costs_, x0, tol=1e-8,
                      alpha_fn=alpha_fn, alpha_gran=100)

