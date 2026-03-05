# Projection method
This repository implements the Projection method to solve the optimal routing problem
specified in Section 5.7.3 of the *Data Networks book of Bertsekas* (see reference below).

The problem is the following:
```math
\begin{align} \min_x & \sum_{(i,j)}D_{ij}(F_{ij})\\ s.t. & \sum_{p\in P_w} x_p = r_w,\quad \forall w \\ & x_p\geq0,\quad \forall p\in P_w, w \end{align}
```
with $x=<x_p>$ the flow vector, $x_p$ the flow traversing path $p$, $(i,j)$ the links of a graph, $D_{ij}(x)$ the cost function when $x$ flow units traverse $(i,j)$, $w$ an origin-destination (OD), $r_w$ the flow requirement of OD $w$, $P_w$ a set of paths that OD $w$ can take, and $F_{ij}=\sum_{w,p\in P_w: (i,j)\in p} x_p$ the total flow traversing link $(i,j)$.

The function $D_{i,j}$ is positive, convex and differentiable.

The repository also supports the equivalent case in which the cost functions
correspond to graph nodes, i.e. $D_n$ with $n$ a graph node.

```bibtex
@book{bertsekas2021data,
  title={Data networks},
  author={Bertsekas, Dimitri and Gallager, Robert},
  year={2021},
  publisher={Athena Scientific}
}
```

