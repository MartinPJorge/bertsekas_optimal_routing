import networkx as nx
import numpy as np
import time

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
            w: {p for p,path in enumerate(paths) if en in path\
                    or en in [el for item in path for el in item]}
            for w,paths in od_paths.items()
        }
        for en in costs.keys()
    }

    # Flow traversing an element (edge or node) for solution x
    F = lambda en, x: sum([x[w][p] for w in od_paths.keys()\
                        for p in cross[en][w]])

    # Total cost of solution x
    total = lambda x: sum([costs[en](F(en,x)) for en in costs.keys()])

    # Elements (edge or node) on a the p^th path of OD w
    elements = lambda w, p: {en for en,wp in cross.items() if p in wp[w]}

    # dp of [(5.84),1]: first derivative p^th path of OD w @solution x
    dp = lambda w, p, x: sum([costs_[en](F(en,x)) for en in elements(w,p)])

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







def flow_deviation(G, od_paths, od_req, costs, costs_, x0, tol,
                   alpha_fn=None, alpha_gran=100, debug=True,
                   alpha_kwargs=None):
    """
    Executes the flow deviation method described in [5.6.1, 1].
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
    :param x0: initial solution dictionary {OD: [F1,F2,...]}
    :param tol: tolerance stopping criteria, if 1-f(xk+1)/f(xk)<=tol stop
    :param alpha_fn: function f(xp,x_,D,D_,alpha_gran,*alpha_kwargs)
                     to obtain the best step alpha with D,D_ the first/second
                     derivative cost functions, and *alpha_kwargs a list of
                     optional arguments
    :param alpha_gran: granularity to find the best step size (used if
                        alpha_fn=None)
    :param alpha_kwargs: list of optional argumens for the alpha_fn
    :param debug: boolean to print or not the progress
    :return: a dictionary specifying the ammount of flow sent over each OD path
                {OD: [F1,F2,...]}
    """

    # Create the dictionary specifying the OD and paths crossing each node/edge
    tic = time.process_time()
    cross = {
        en : {
            w: {p for p,path in enumerate(paths) if en in path\
                    or en in [el for item in path for el in item]}
            for w,paths in od_paths.items()
        }
        for en in costs.keys()
    }
    tac = time.process_time()
    if debug: print('Time computing en crossings: ', tac-tic)

    # Elements (edge or node) on p^th path of OD w
    elements = {
        w: {
            p: {en for en,wp in cross.items() if p in wp[w]}
            for p,path in enumerate(paths)
        }
        for w,paths in od_paths.items()
    }

    # Flow traversing an element (edge or node) for solution x
    F = lambda en, x: sum([x[w][p] for w in od_paths.keys()\
                        for p in cross[en][w]])

    # Total cost of solution x
    total = lambda x: sum([costs[en](F(en,x)) for en in costs.keys()])

    # Total cost derivative of solution x
    total_ = lambda x: sum([costs_[en](F(en,x)) for en in costs_.keys()])

    # # Elements (edge or node) on a the p^th path of OD w
    # elements = lambda w, p: {en for en,wp in cross.items() if p in wp[w]}

    # dp of [(5.84),1]: first derivative p^th path of OD w @solution x
    #dp = lambda w, p, x: sum([costs_[en](F(en,x)) for en in elements(w,p)])
    dp = lambda w, p, x: sum([costs_[en](F(en,x)) for en in elements[w][p]])

    # flow deviation step [(5.62),1]: given the current solution xp and the
    #                                 MFDL solution x_ w/ step alpha
    fd = lambda xp,x_,alpha: {
        w: [xp[w][p]+alpha*(x_[w][p]-xp[w][p]) for p,path in enumerate(paths)]
        for w,paths in od_paths.items()
    }


    # Set large solution (2x initial) and initial
    xk  = {w: [2*xp for xp in x0[w]] for w in x0.keys()}
    xk1 = x0
    # Projection method loop [(5.83)-(5.85),1]
    while 1 - total(xk1) / total(xk) > tol:
        xk = xk1

        # minimum first derivative length/cost (MFDL) path of each OD pair w
        # @solution xk
        tic = time.process_time()
        mfdlp = {
            w: int(np.argmin([dp(w,p,xk) for p,path in enumerate(paths)]))
            for w,paths in od_paths.items()
        }
        tac = time.process_time()
        if debug: print(' Time computing MDLP:', tac-tic)

        # route all OD traffic along the MFDL path
        x_ = {
            w: [0 if p!= mfdlp[w] else od_req[w] for p,_ in enumerate(paths)]
            for w,paths in x0.items()
        }

        # Do the flow deviation step
        tic = time.process_time()
        if alpha_fn != None:
            xk1 = fd(xk,x_,alpha_fn(xk,x_,total,total_,alpha_gran,
                                    *(alpha_kwargs+[F])))
        else:
            # Line search over alpha to obtain the minimum cost
            alpha_ = 0
            xk1 = fd(xk,x_,alpha_)
            for alpha_ in np.linspace(0, 1, alpha_gran):
                x_fd = fd(xk1,x_,alpha_)
                xk1 = x_fd if total(x_fd) < total(xk1) else xk1
        tac = time.process_time()
        if debug: print(' Time computing best alpha:', tac-tic)

        if debug: print(f'D={total(xk1)}')

    return xk1





def flow_deviation_v(G, od_paths, od_req, costs, costs_, x0, tol,
                   alpha_fn=None, alpha_gran=100, debug=True,
                   alpha_kwargs=None, cross=None, elements=None, phi=None):
    """
    Executes the flow deviation method described in [5.6.1, 1].
    Each origin-destination (OD) pair has a list of possible paths and a
    flow requirement.
    The goal is to find the optimal routing to minimize the cost of
    each link.
    This version uses each solution as a np.array to speed up the execution.

    [1] Bertsekas, D., & Gallager, R. (2021). Data networks. Athena Scientific.

    :param G: a networkx graph
    :param od_paths: dictionary of OD paths {OD: [p1,p2,...]}, each path is a
                    list of edges of G
    :param od_req: OD flow requirement
    :param costs: dictionary of cost functions, of edges or nodes
                    {e: f(Fe)} or {n: f(Fn)}, w/ e an edge and n a node
    :param costs_: dictionary of cost functions' derivatives, of edges or nodes
                    {e: f'(Fe)} or {n: f'(Fn)}, w/ e an edge and n a node
    :param x0: initial solution dictionary {OD: [F1,F2,...]}
    :param tol: tolerance stopping criteria, if 1-f(xk+1)/f(xk)<=tol stop
    :param alpha_fn: function f(xp,x_,D,D_,alpha_gran,*alpha_kwargs)
                     to obtain the best step alpha with D,D_ the first/second
                     derivative cost functions, and *alpha_kwargs a list of
                     optional arguments
    :param alpha_gran: granularity to find the best step size (used if
                        alpha_fn=None)
    :param alpha_kwargs: list of optional argumens for the alpha_fn
    :param debug: boolean to print or not the progress
    :param cross: a dictionary specifying the OD and paths crossing each edge/node
    :param elements: a dictionary specifying the edge/node on p^th path of OD w
    :param phi: a dictionary with edge/node keys
                phi[en] = np.array(size=k·p) w/ 1 if en in (k,p)
    :return: a dictionary specifying the ammount of flow sent over each OD path
                {OD: [F1,F2,...]}
             a dictionary specifying the OD and paths crossing each edge/node
             a dictionary specifying the edge/node on p^th path of OD w
             a dictionary with edge/node keys
                phi[en] = np.array(size=k·p) w/ 1 if en in (k,p)
    """

    if cross == None:
        # Create the dictionary specifying the OD and paths crossing each
        # node/edge
        tic = time.process_time()
        cross = {
            en : {
                w: {p for p,path in enumerate(paths) if en in path\
                        or en in [el for item in path for el in item]}
                for w,paths in od_paths.items()
            }
            for en in costs.keys()
        }
        tac = time.process_time()
        if debug: print('Time computing en crossings: ', tac-tic)

    if elements == None:
        # Elements (edge or node) on p^th path of OD w
        elements = {
            w: {
                p: {en for en,wp in cross.items() if p in wp[w]}
                for p,path in enumerate(paths)
            }
            for w,paths in od_paths.items()
        }

    if phi == None:
        # phi[en] = np.array(size=k·p) w/ 1 if en in (k,p)
        phi = {
            en: np.array([1 if en in elements[w][p] else 0
                    for w,paths in od_paths.items()\
                    for p,path in enumerate(paths)])
            for en in costs.keys()
        }

    # Transform x {OD:[p1,p2,...]} dictionary to np.array
    x2v = lambda x: np.array([xkp for xk in x.values() for xkp in xk])

    # Flow traversing an element (edge or node) for solution verctor xv
    Fv = lambda en, xv: xv.dot(phi[en])

    # Total cost of solution vector x
    totalv = lambda xv: sum([costs[en](Fv(en,xv)) for en in costs.keys()])

    # Total cost derivative of solution vector x
    total_v = lambda xv: sum([costs_[en](Fv(en,xv)) for en in costs_.keys()])

    # dp of [(5.84),1]: first derivative p^th path of OD w @solution vec xv
    dpv = lambda w, p, xv: sum([costs_[en](Fv(en,xv)) for en in elements[w][p]])



    # Set large solution (2x initial) and initial
    x0v = x2v(x0) # convert to numpy array the initial solution
    xk1v = x0v
    xkv = 2*x0v # set large initial solution
    tot_xk1v, tot_xkv = totalv(xk1v), totalv(xkv)
    # Projection method loop [(5.83)-(5.85),1]
    while 1 - tot_xk1v / tot_xkv > tol:
        xkv, tot_xkv = xk1v, tot_xk1v

        # minimum first derivative length/cost (MFDL) path of each OD pair w
        # @solution xk
        tic = time.process_time()
        mfdlp = {
            w: int(np.argmin([dpv(w,p,xkv) for p,path in enumerate(paths)]))
            for w,paths in od_paths.items()
        }
        tac = time.process_time()
        if debug: print(' Time computing MDLP:', tac-tic)

        # route all OD traffic along the MFDL path
        x_v = np.array([
            0 if p!= mfdlp[w] else od_req[w]\
            for w,paths in x0.items()\
            for p,_ in enumerate(paths)
        ])

        # Do the flow deviation step
        tic = time.process_time()
        alpha_fn = None
        if alpha_fn != None:
            xk1 = fd(xk,x_,alpha_fn(xk,x_,total,total_,alpha_gran,
                                    *(alpha_kwargs+[F])))
        else:
            # Line search over alpha to obtain the minimum cost
            alpha_ = 0
            xk1v = xkv + alpha_*(x_v-xkv) 
            best_total = totalv(xk1v)
            for alpha_ in np.linspace(0, 1, alpha_gran):
                # flow deviation step [(5.62),1]: given the current solution xp
                #                                 and the MFDL solution x_ w/
                #                                 step alpha
                x_fdv = xkv + alpha_*(x_v-xkv)
                tot = totalv(x_fdv)
                if tot < best_total:
                    xk1v, best_total = x_fdv, tot
        tac = time.process_time()
        if debug: print(' Time computing best alpha:', tac-tic)

        if debug: print(f'D={totalv(xk1v)}')

        # Keep track of the total cost of the obtained solution
        tot_xk1v = best_total


    # Transform the solution vec xk1v into a dict {OD:[p1,p2,...]}
    xk1, idx = {w: [] for w in od_paths.keys()}, 0
    for w,paths in od_paths.items():
        for p,path in enumerate(paths):
            xk1[w].append(xk1v[idx])
            idx += 1

    return xk1, cross, elements, phi





