import numpy as np
import itertools

A = np.array([[0,15,8,21],
              [15,0,13,14],
              [8,13,0,20],
              [21,14,20,0]])


def tuple_remove(T,i):
    L = list(T)
    L.remove(i)

    return tuple(L)



def Held_Karp(A):
    N = A.shape[0]
    g = {}
    Sets = set(range(1,N))
    for j in range(1,N):
        g[((j,),j)] = A[0,j]

    for s in range(2,N):
        subsets = itertools.combinations(Sets,s)

        for S in subsets:
            for k in S:
                g[(S,k)]= min([g[(tuple_remove(S,k),m)] + A[m,k] for m in S if m != k])

        
    V = min([g[(tuple(range(1,N)),k)] + A[k,0] for k in range(1,N)])
    return V


print(Held_Karp(A))
