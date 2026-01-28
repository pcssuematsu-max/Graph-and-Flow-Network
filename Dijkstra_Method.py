import numpy as np
import heapq
inf = float("inf")

Graph = {("A","B"):7,
         ("A","E"):9,
         ("A","F"):14,
         ("B","C"):8,
         ("C","D"):6,
         ("C","E"):10,
         ("D","F"):4,
         ("E","F"):7}

Vertices = ["A","B","C","D","E","F"]


def dijkstra(G,V,start):
    Dist = {v:inf for v in V}
    Prev = {v:None for v in V}
    Dist[start] = 0

    Q = [(0,start)]
    visited = set([])

    while Q:
        d,u = heapq.heappop(Q)
        if u in visited:
            continue
        visited.add(u)

        for (a,b),cost in G.items():
            if a == u and b not in visited:
                alt = d + cost
                if alt < Dist[b]:
                    Dist[b] = alt
                    Prev[b] = u
                    heapq.heappush(Q,(alt,b))
            elif b == u and a not in visited:
                alt = d + cost
                if alt < Dist[a]:
                    Dist[a] = alt
                    Prev[a] = u
                    heapq.heappush(Q,(alt,a))

    return Dist,Prev
        


    return Dist,Prev

print(dijkstra(Graph,Vertices,"A"))
