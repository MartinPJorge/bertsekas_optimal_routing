import networkx as nx
import numpy as np

def projection_method(G, od_paths, od_req, costs, costs_, costs__, x0, tol,
                      alpha):
    """
    Executes the projection method described in [5.7.3, 1].
    Each origin-destination (OD) pair has a list of possible paths and a
    flow requirement.
    The goal is to find the optimal routing to minimize the cost of
    each link.

    [1] Bertsekas, D., & Gallager, R. (2021). Data networks. Athena Scientific.

    :param G: a networkx graph
    :param od_paths: dictionary of OD paths {OD: [p1,p2,...]}, each path is a
                    list of edges of G
    :param od_req: OD flow requirement
    :param costs: dictionary of cost functions, of edges or nodes
                    {e: f(Fe)} or {n: f(Fn)}, w/ e an edge and n a node
    :param costs_: dictionary of cost functions' derivatives, of edges or nodes
                    {e: f'(Fe)} or {n: f'(Fn)}, w/ e an edge and n a node
    :param costs__: dictionary of cost functions' 2nd derivatives, of edges or
                    nodes {e: f'(Fe)} or {n: f'(Fn)}
    :param x0: initial solution dictionary {OD: [F1,F2,...]}
    :param tol: tolerance stopping criteria, if |f(xk)-f(xk+1)|<tol stop
    :param alpha: the step size at each iteration
    :return: a dictionary specifying the ammount of flow sent over each OD path
                {OD: [F1,F2,...]}
    """

    # Initalize the set of paths with zero flow
    zero_paths = {w: [p for p,F in enumerate(x0[w]) if F==0]\
                    for w in od_paths.keys()}

    # Create the dictionary specifying the OD and paths crossing each node/edge
    cross = {
        en : {
            w: {p for p,path in enumerate(paths) if ne in path\
                    or ne in [el for el in item for item in path]}
            for w,paths in od_paths.items()
        }
        for en in costs.keys()
    }

    # Flow traversing an element (edge or node) for solution x
    F = lambda en, x: sum([x[w][p] for w in od_paths.keys()\
                        for p in cross[en][w]])

    # Total cost of solution x
    total = lambda x: sum([cost[en](F(en,x)) for en in cost.keys()])

    # Elements (edge or node) on a the p^th path of OD w
    elements = lambda w, p: {en for en,wp in cross.items() if p in wp[w]}

    # dp of [(5.84),1]: first derivative p^th path of OD w @solution x
    dp = lambda w, p, x: sum([cost_(F(en,x)) for en in elements(w,p)])

    # Lp: elements (edge or nodes) within p^th path of OD w,
    #     or the MFDL path p_^th of OD w, but not both
    Lp = lambda w, p, p_: elements(w,p).difference(elements(w,p_))

    # Hp of [(5.85),1]: second derivative p^th path of OD w @solution x
    #                   knowing the MDFL path p_^th of OD w
    Hp = lambda w, p, p_, x: sum([cost__(F(en,x)) for en in Lp(w,p,p_)])


    # Set large solution (2x initial) and initial
    xk  = {w: [2*xp for xp in x0[w]] for w in x0.keys()}
    xk1 = x0
    k = 0
    # Projection method loop [(5.83)-(5.85),1]
    while abs(total(xk) - total(xk1)) < tol:
        xk = xk1

        # minimum first derivative length/cost (MFDL) path of each OD pair w
        # @solution xk
        mfdlp = {
            w: int(np.argmin([dp(w,p,xk) for p,path in enumerate(paths)]))
            for w,paths in od_paths.items()
        }

        # Use (5.83): update the flow of paths carrying non-zero flow
        for w,paths in od_paths.items():
            dp_ = dp(w, mdflp[w], xk) # MDFL
            for p,path in enumerate(paths):
                if p in zero_paths[w]:
                    continue
                xk1[w][p] = max(0, xk[w][p]\
                    - alpha**k / Hp(w,p,mdfl[p],xk) * (dp(w,p,xk) - dp_))

            # Update the list of paths with zero flow
            if xk1[w][p] == 0:
                zero_paths[w].append(p)

        k += 1

    return xk1

