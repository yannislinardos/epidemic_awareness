import numpy as np
import networkx as nx


# m: mode
# a: shape
def inverse_cdf_pareto(tau, m=1):
    a = tau - 1
    f = lambda u: m / ((1 - u) ** (1 / a))
    f_vec = np.vectorize(f)
    return f_vec


# My implementation of the pareto distribution
def get_pareto_sequence(tau, n, m=1, cutoff=False):
    f = inverse_cdf_pareto(tau, m)

    w = np.sort(f(np.random.uniform(0, 1, n))).astype(np.int)
    while np.sum(w) % 2 != 0:
        w = np.sort(f(np.random.uniform(0, 1, n))).astype(np.int)

        if cutoff:
            low, high = cutoff
            w[w < low] = low
            w[w > high] = high

    return w


def delete_self_loops(G):

    self_loops = list(nx.classes.function.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    return G


def get_self_loops(G):
    return list(nx.classes.function.selfloop_edges(G))


def get_multiple_edges(G):
    ret = []
    for u in G.nodes():
        for neighbor in G.neighbors(u):
            if G.number_of_edges(u, neighbor) > 2:
                if (u, neighbor) not in ret and (neighbor, u) not in ret:
                    ret.append((u, neighbor))
    return ret


def remove_mult_edges(G):

    mult_edges = get_multiple_edges(G)
    G.remove_edges_from(mult_edges)

    return G


def make_simple_graph(G):
    G = remove_mult_edges(G)
    G = delete_self_loops(G)
    return G


def get_fractions_from_degree_dist(d):

    nodes = d.size
    max_degree = np.max(d)
    p = np.zeros(max_degree+1)
    unique, counts = np.unique(d, return_counts=True)
    p[unique] = counts
    p = p/nodes
    return p

