import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import numpy as np

def construct_arbitrage_graph(securities, exchange_rates):
    G = nx.DiGraph()
    G.add_nodes_from(securities)
    G.add_weighted_edges_from(exchange_rates)
    return G

def edges_from_node(G, n):
    """
        n is the index of node in G.nodes
        res is an array containing the index of edge in G.edges or key of edges
    """
    edges = list(G.edges)
    nodes = list(G.nodes)
    var = [i for i in range(len(G.edges))]
    edge2var = {edges[i]:i for i in var}
    res = []
    
    for e in edges:
        if e[0]==nodes[n]: res.append(edge2var[e])
    return res

def edges_to_node(G, n):
    """
        n is the index of node in G.nodes
        res is an array containing the index of edge in G.edges or key of edges
    """
    edges = list(G.edges)
    nodes = list(G.nodes)
    var = [i for i in range(len(G.edges))]
    edge2var = {edges[i]:i for i in var}
    res = []
    for e in edges:
        if e[1]==nodes[n]: res.append(edge2var[e])
    return res

def check_constraint(G, sol):
    edges = list(G.edges.data())
    nodes = list(G.nodes)
    var = [i for i in range(len(edges))]

    conservation_constraint = 0
    for v in range(len(nodes)):
        edges_from_v = edges_from_node(G, v)
        edges_to_v = edges_to_node(G, v)
        difference = [sol[e] for e in edges_from_v]+[-sol[e] for e in edges_to_v]
        conservation_constraint += abs(sum(difference))
    #print("Violated conservation constraint by this much:", conservation_constraint)

    one_pass_constraint = 0
    for v in range(len(nodes)):
        edges_from_v = edges_from_node(G, v)
        for e1 in edges_from_v:
            tmp = 0
            for e2 in edges_from_v:
                tmp += sol[e2]
            one_pass_constraint += (sol[e1]*(tmp-1))
    #print("Violated one pass constraint by this much:", one_pass_constraint)

    return conservation_constraint, one_pass_constraint

def calc_profit(sol):
    profit = 1
    for s in sol:
        profit *= s[2]['weight']
    return profit

def get_route(sol):
    route = []
    for s in sol:
        route.append((s[0],s[1]))
    return route

def construct_qubo(G):
    edges = list(G.edges.data())
    print(edges)
    nodes = list(G.nodes)
    var = [i for i in range(len(edges))]
    Q = np.zeros((len(var), len(var)))


    #TODO: this is added in by me manually to achieve the paper's result
    M_europe = 10e10
    index_europe = nodes.index('EUR')
    edges_from_v = edges_from_node(G, index_europe)
    edges_to_v = edges_to_node(G, index_europe)
    # Mx
    for e in edges_to_v:
        Q[e][e] += M_europe
    for e in edges_from_v:
        Q[e][e] += M_europe

    M1 = 1
    # objective function coefficient
    for i in var:
        edge_weight = edges[i][2]['weight']
        Q[i][i] += -np.log(edge_weight)*M1
    #print(Q)

    # flow conservation constraint
    # M has to be large otherwise a cycle may not form 
    # (the algorithm might think that if I only pick an edge with a super high value it would fare better than forming a cycle)
    # numerically that is true but we need a cycle! so we need to regularize accordingly
    M2 = 1000
    for v in range(len(nodes)):
        edges_from_v = edges_from_node(G, v)
        edges_to_v = edges_to_node(G, v)
        difference = [(1,e) for e in edges_from_v]+[(-1,e) for e in edges_to_v]
        for i1,d1 in enumerate(difference):
            for i2,d2 in enumerate(difference):
                coeff1, coeff2 = d1[0], d2[0]
                edge_index_1, edge_index_2 = d1[1], d2[1]
                if edge_index_1<edge_index_2:
                    # to be consistent with the upper tri form of QUBO matrix
                    Q[edge_index_1][edge_index_2] += M2*(coeff1*coeff2)
                else:
                    Q[edge_index_2][edge_index_1] += M2*(coeff1*coeff2)
    #print(Q)
    # one pass constraint
    M3 = 500
    for v in range(len(nodes)):
        edges_from_v = edges_from_node(G, v)

        for e1 in edges_from_v:
            for e2 in edges_from_v:
                if e1<e2:
                    Q[e1][e2] += M3
                else:
                    Q[e2][e1] += M3
            Q[e1][e1] -= M3
    
    #print(Q)
    assert np.allclose(Q, np.triu(Q)) # check if upper triangular
    return Q, edges

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DRAW = False


    securities = ['JPY','EUR','CAD','CNY','USD']
    exchange_rates = [
        ('JPY','EUR',0.00872),
        ('EUR','JPY',114.65000),
        ('JPY','CAD',0.01266),
        ('CAD','JPY',78.94000),
        ('JPY','USD',0.00961),
        ('USD','JPY',104.05000),
        ('JPY','CNY',0.06463), 
        ('CNY','JPY',15.47000), #JPY Done
        ('CNY','EUR',0.13488),
        ('EUR','CNY',7.41088),
        ('CNY','CAD',0.19586),
        ('CAD','CNY',5.10327),
        ('CNY','USD',0.14864),
        ('USD','CNY',6.72585), #CNY Done
        ('USD','EUR',0.90745),
        ('EUR','USD',1.10185),
        ('USD','CAD',1.31904),
        ('CAD','USD',0.75799), #USD Done
        ('CAD','EUR',0.68853),
        ('EUR','CAD',1.45193), #CAD Done
    ]

    G = construct_arbitrage_graph(securities, exchange_rates)
    edge_weights = nx.get_edge_attributes(G,'weight')
    if DRAW:
        nx.draw(G, pos=nx.circular_layout(G), with_labels=True, connectionstyle='arc3, rad = 0.1')
        #nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=edge_weights)
        plt.show()

    Q, edges = construct_qubo(G)
    #print(Q)
    sampler = DWaveSampler()
    #print(sampler.properties['h_range'])
    #print(sampler.properties['j_range'])
    sampler_embedded = EmbeddingComposite(sampler)
    sampleset = sampler_embedded.sample_qubo(Q, num_reads=10000)


    sample_result = sampleset.record.tolist()
    sample_result.sort(key = lambda s:s[1]) # sort based on energy
    #print(sampleset)
    

    """for r,sol in enumerate(sample_result):
        picked = []
        zeroed = []
        for i,e in enumerate(sol[0]):
            if e == 1:
                picked.append(edges[i])
                zeroed.append(1)
            else:
                zeroed.append(0)
        c1, c2 = check_constraint(G, zeroed)
        if c1 == 0 and c2 == 0:
            print("Valid cycle:", picked)"""
    print("Top results that satisfy the flow conservation constraint")
    champions = []
    cnt = 1
    for sol in [i[0] for i in sample_result]:
        picked = []
        zeroed = []
        for i,e in enumerate(sol):
            if e == 1:
                picked.append(edges[i])
                zeroed.append(1)
            else:
                zeroed.append(0)
        conservation_constraint, one_pass_constraint = check_constraint(G, zeroed)
        if conservation_constraint == 0:
            profit = calc_profit(picked)
            route = get_route(picked)
            if profit > 1:
                print("Rank {} result is formed by {} with profit {}".format(cnt, route, profit))
        cnt += 1