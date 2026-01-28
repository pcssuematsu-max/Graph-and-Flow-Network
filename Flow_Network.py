import numpy as np
import heapq
vertices = ["S","A","B","C","D","T"]
capacities = {("S","A"):10,
              ("S","C"):10,
              ("A","B"):4,
              ("A","C"):2,
              ("A","D"):8,
              ("B","T"):10,
              ("C","D"):9,
              ("D","B"):6,
              ("D","T"):10}


mat = np.array([[1,6,2,3,7,5,4],
                [6,2,5,3,4,1,7],
                [4,1,2,3,7,6,5],
                [1,3,5,7,2,4,6],
                [2,5,1,3,6,4,7],
                [7,2,3,4,1,5,6],
                [5,2,7,6,1,4,3]])


def hungarian_method(mat):
    """
    Solve the assignment problem by Hungarian method
    """
    mat_B = mat.copy()
    NIL = mat.shape[1]

    #Step 1    
    mat_B -= np.min(mat_B,axis = 1).reshape(-1,1)
    mat_B -= np.min(mat_B,axis = 0)

    #Step 2
    mat_zero = np.zeros_like(mat_B,dtype = "i")
    mat_zero[mat_B == 0] = 1

    #Step 3
    pair_U,pair_V = hopcroft_karp(mat_zero)
    keep_searching = True
    
    while keep_searching:
        if np.all(pair_U != NIL):
            keep_searching = False
        else:
            U_searched,V_searched = bfs_for_hungarian_method(mat_zero,pair_U,pair_V,NIL)
            U_unsearched = ~U_searched
            V_unsearched = ~V_searched

            mask_A = U_searched.reshape(-1,1) @ V_unsearched.reshape(1,-1)
            mask_B = U_unsearched.reshape(-1,1) @ V_searched.reshape(1,-1)
            
            M = np.min(mat_B[mask_A])
            mat_B[mask_A] -= M
            mat_B[mask_B] += M
            
            
            mat_zero = np.zeros_like(mat_B,dtype = "i")
            mat_zero[mat_B == 0] = 1
            pair_U,pair_V = hopcroft_karp(mat_zero)

    return pair_U,pair_V

        
def bfs_for_hungarian_method(mat,pair_U,pair_V,NIL):
    """
    Breadth-first search for Step 3 of Hungarian Method.
    """
    searching = [u for u in range(pair_U.shape[0]) if pair_U[u] == NIL]
    

    U_searched = (pair_U == NIL)
    V_searched = np.zeros_like(pair_V,dtype = bool)
    
    k = 0
    while k < len(searching):
        u = searching[k]
        for v in range(pair_V.shape[0]):
            if mat[u,v] == 1 and pair_U[u] != v:
                V_searched[v] = True
                u2 = int(pair_V[v])
                if not U_searched[u2]:
                    searching.append(u2)
                    U_searched[u2] = True

        k += 1

    return U_searched,V_searched

    

def hopcroft_karp(mat):
    """
    Solve the maximum-cardinality matching problem by Hopcroft-Karp algorithm.
    """ 
    NIL_U,NIL_V = mat.shape
    INF_U,INF_V = NIL_U,NIL_V
    
    pair_U = NIL_V * np.ones(mat.shape[0],dtype = "i")
    pair_V = NIL_U * np.ones(mat.shape[1],dtype = "i")
    
    
    keep_bfs = True
    matched = 0
    while keep_bfs:
        level = bfs_for_hopcroft_karp(mat,pair_U,pair_V,NIL_U,NIL_V,INF_U,INF_V)
        keep_bfs = (level[NIL_U] < INF_U)
        if keep_bfs:
            for u in range(mat.shape[0]):
                if pair_U[u] == NIL_V:
                    if dfs_for_hopcroft_karp(u,mat,pair_U,pair_V,level,NIL_U,NIL_V,INF_U,INF_V):
                        matched += 1

    return pair_U,pair_V            


def dfs_for_hopcroft_karp(u,mat,pair_U,pair_V,level,NIL_U,NIL_V,INF_U,INF_V):
    """
    Depth-first search for Hopcroft-Karp algorithm.
    """
    if u == NIL_U:
        return True

    for v in range(mat.shape[1]):
        if mat[u,v] == 1 and v != pair_U[u]:
            u2 = pair_V[v]
            if level[u2] == level[u] + 1:
                if dfs_for_hopcroft_karp(u2,mat,pair_U,pair_V,level,NIL_U,NIL_V,INF_U,INF_V):
                    pair_U[u] = v
                    pair_V[v] = u
                    return True

    level[u] = INF_V
    return False


    
def bfs_for_hopcroft_karp(mat,pair_U,pair_V,NIL_U,NIL_V,INF_U,INF_V):
    """
    Breadth-first search for Hopcroft-Karp algorithm.
    """
    searching = [i for i in range(mat.shape[0]) if pair_U[i] == NIL_V]
    level = INF_V * np.ones(NIL_U + 1,dtype = "i")
    for i in searching:
        level[i] = 0

    k = 0
    while k < len(searching):
        u = searching[k]
        if level[u] < level[NIL_U]:
            for v in range(mat.shape[1]):
                if mat[u][v] == 1 and v != pair_U[u]:
                    u2 = pair_V[v]
                    if level[u2] == INF_V:
                        level[u2] = level[u] + 1
                        searching.append(u2)

        k += 1




    return level



def dinic(vertices,capacities,start,terminal):
    """
    Solve the maximum flow problem by Dinic algorithm.
    """
    vertices_n = len(vertices)
    mat = make_matrix(vertices,capacities)
    flow = np.zeros_like(mat,dtype = "f")
    maximum_flow = 0
    start_i= vertices.index(start)
    terminal_i = vertices.index(terminal)
    keep_searching = True
    residual_flow = mat.copy()
    while keep_searching:
        level = breadth_first_search(residual_flow,start_i,terminal_i)[1]
        keep_searching = (level[terminal_i] != vertices_n)
        ptr = np.zeros(vertices_n,dtype = "i")
        keep_dfs = True
        while keep_dfs:
            f = depth_first_search(residual_flow,np.inf,start_i,terminal_i,level,vertices_n,ptr)
            maximum_flow += f
            keep_dfs = (f > 0)

    flow = mat - residual_flow
    return residual_flow,flow
                                
        
    
def depth_first_search(mat,current_flow,u,terminal_i,level,vertices_n,ptr):
    if u == terminal_i:
        return current_flow

    for v in range(ptr[u],vertices_n):
        ptr[u] = v
        if level[v] == level[u] + 1 and mat[u,v] > 0:
            pushed = depth_first_search(mat,min(current_flow,mat[u,v]),v,terminal_i,level,vertices_n)
            if pushed > 0:
                mat[u,v] -= pushed
                mat[v,u] += pushed
                return pushed


    return 0
    


def edmonds_karp(vertices,edges,start,terminal):
    """
    Solve the maximum flow problem by Edmonds-Karp algorithm. 
    """
    mat = make_matrix(vertices,edges)
    flow = np.zeros_like(mat,dtype = "f")
    maximum_flow = 0
    start_i= vertices.index(start)
    terminal_i = vertices.index(terminal)

    keep_searching = True
    while keep_searching:
        residual_flow = mat - flow
        path,level = breadth_first_search(residual_flow,start_i,terminal_i)
        if path is None:
            keep_searching = False
        else:
            values = [residual_flow[path[i],path[i + 1]] for i in range(len(path) - 1)]
            max_value = np.min(values)
            maximum_flow += max_value
            for i in range(len(path) - 1):
               flow[path[i],path[i+1]] += max_value
               flow[path[i+1],path[i]] -= max_value

    return maximum_flow,flow,mat - flow

def dijkstra(mat,start_i,terminal_i):
    """
    Find one of the shortest path from the start to the terminal.
    """
    size = len(vertices)
    distance = np.inf * np.ones(size,dtype = "f")
    previous = [None] * size    
    distance[start_i] = 0

    Q = [(0,start_i)]
    visited = set([])

    while Q:
        (d,u) = heapq.heappop(Q)
        if u in visited:
            continue
        visited.add(u)

        for b in range(size):
            if b not in visited:
                cost = mat[u,b]
                alt = d + cost
                if alt < distance[b]:
                    distance[b] = alt
                    previous[b] = u
                    heapq.heappush(Q,(alt,b))


    if distance[terminal_i] == np.inf:
        return (None,np.inf)
    else:
        path = (terminal_i,)
        u = terminal_i
        while u != start_i:
            u = previous[u]
            path += (u,)

        
        return (path[::-1],distance)



def successive_shortest_path(vertices,capacities,costs,flow,start,terminal):
    """
    Solve minimum-cost flow problem by successive_shortest_path.
    """
    size = len(vertices)
    potential = np.zeros(size,dtype = "f")
    current_flow = 0.0
    mat_capacities = make_matrix(vertices,capacities)
    mat_costs_original = make_matrix(vertices,costs,costs_mode = True)
    mat_flow = np.zeros_like(mat_capacities,dtype = "f")
    residual_flow = mat_capacities - mat_flow

    mat_costs = mat_costs_original.copy()
    mat_costs[residual_flow < 1.0e-8] = np.inf
    
    start_i = vertices.index(start)
    terminal_i = vertices.index(terminal)
    current_flow = 0
    current_cost = 0
    keep_searching = True
    while keep_searching:
        path,distance = dijkstra(mat_costs,start_i,terminal_i)
        if path is None:
            print('Path is None.')
            return None
        potential -= distance
        print(potential)
        v = 0
        max_flow = min([residual_flow[path[i],path[i+1]] for i in range(len(path) - 1)] + [flow - current_flow])
        current_flow += max_flow
        current_cost += distance[terminal_i] * max_flow

     
        if current_flow == flow:
            keep_searching = False

        
        for i in range(len(path) - 1):
            mat_flow[path[i],path[i+1]] += max_flow
            mat_flow[path[i+1],path[i]] -= max_flow

        residual_flow = mat_capacities - mat_flow
        mat_costs = mat_costs_original.copy()
        mat_costs[residual_flow < 1.0e-8] = np.inf
        mat_costs -= potential.reshape(-1,1)
        mat_costs += potential




    
    
    return mat_flow,current_cost


        




    

def make_matrix(vertices,edges,costs_mode = False):
    """
    Make the flow matrix from the given graph.
    """
    n = len(vertices)
    indices = {vertices[i]:i for i in range(n)}
    mat = np.zeros((n,n),dtype = "f")
    for e in edges:
        mat[indices[e[0]],indices[e[1]]] = edges[e]
        if costs_mode:
            mat[indices[e[1]],indices[e[0]]] = -edges[e]

    return mat

def breadth_first_search(mat,start_i,terminal_i):
    """
    Find the shortest path from the graph represented by the matrix mat.
    """

    vertices_n = mat.shape[0]
    searching = [start_i]
    paths = {start_i:(start_i,)}
    level = (vertices_n) * np.ones(vertices_n,dtype = "i")
    level[start_i] = 0

    k = 0
    keep_searching = True
    while k < len(searching) and keep_searching:
        i = searching[k]
        for j in range(vertices_n):
            if mat[i,j] > 0:
                if level[j] > level[i] + 1:
                    level[j] = level[i] + 1
                    paths[j] = paths[i] + (j,)         
                    
                if j in searching:
                    continue
                searching.append(j)

        k += 1


    if terminal_i in paths:
        return (paths[terminal_i],level)
    else:
        return (None,level)


print(hungarian_method(mat))

