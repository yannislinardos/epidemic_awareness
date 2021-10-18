import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain
from networkx.generators.degree_seq import _to_stublist as to_stublist
from networkx.algorithms.operators.binary import compose
from utils import *


class HierarchicalConfigurationModel:
    """
    Undirected Hierarchical Configuration Model
    communities: iterable with len(communities) = # of communities and len(communities[i]) the size of community i
    in_degrees: iterable of iterables with in_degrees[i] being an iterable of the in_degrees of community i
    out_degrees: iterable of the out-degrees of nodes
    """

    def __init__(self, in_degrees, out_degrees):

        if type(in_degrees) is not list and type(in_degrees) is not tuple:
            in_degrees = [in_degrees]

        out_degrees = out_degrees.astype(int)
        for c in range(len(in_degrees)):
            in_degrees[c] = in_degrees[c].astype(int)

        self.in_degrees = in_degrees
        self.out_degrees = out_degrees
        # community lengths
        self.N_H = np.array([len(c) for c in in_degrees])
        # number of nodes
        N_1 = len(out_degrees)
        N_2 = sum([len(c) for c in in_degrees])
        if N_1 != N_2:
            raise Exception('Lengths do not match')
        else:
            self.N = N_1

        # number of communities
        self.n = len(self.N_H)

        # create communities
        self.communities, self.community_per_node, self.real_in_degrees, self.real_in_degrees_per_community\
            = self.create_communities()

        self.nodes_per_community = []
        for _ in range(self.n):
            self.nodes_per_community.append(np.array(self.communities[c].nodes))

        # join communitites
        self.graph = self.join_communities()
        self.graph = make_simple_graph(self.graph)
        self.degrees = np.array(self.graph.degree)[:,1]
        self.real_out_degrees = self.degrees - self.real_in_degrees

        self.p_out = get_fractions_from_degree_dist(self.real_out_degrees)
        max_in = np.max(self.real_in_degrees)
        self.p_in = np.zeros((self.n, max_in + 1))
        for i in range(self.n):
            d = self.real_in_degrees_per_community[i]
            p = get_fractions_from_degree_dist(d)
            self.p_in[i, :p.size] = p

        self.max_k_out = self.p_out.shape[0] - 1
        self.max_k_in = self.p_in.shape[1] - 1

        # fraction of nodes in H with in_degree, and out-degree
        self.p_total = np.zeros((self.n, self.max_k_in+1, self.max_k_out+1))
        for node in range(0, self.N):
            H, k_in, k_out = self.from_node_to_compartment(node)
            self.p_total[H, k_in, k_out] += 1
        self.p_total = self.p_total/self.N

        self.p_out = np.sum(self.p_total, axis=(0,1))
        self.p_in = np.sum(self.p_total, axis=2)
        for com in range(self.n):
            self.p_in[com] = self.p_in[com] *self.N/self.N_H[com]

        self.avg_k_in = np.sum(self.p_in * np.tile(np.arange(0, self.p_in.shape[1]),
                                                   self.p_in.shape[0]).reshape(self.p_in.shape), axis=1)

        self.avg_k_out = np.sum(self.p_out * np.arange(0, self.p_out.shape[0]))

        self.avg_k_out_H = np.zeros(self.n)
        max_out = np.max(self.real_out_degrees)
        self.p_out_H = np.zeros((self.n, max_out + 1))
        for c in range(self.n):
            com = np.array(self.communities[c].nodes)
            self.avg_k_out_H[c] = np.mean(self.real_out_degrees[com])
            d = self.real_out_degrees[com]
            p = get_fractions_from_degree_dist(d)
            self.p_out_H[c, :p.size] = p


    def from_node_to_compartment(self, node):

        H = self.get_community_per_node()[node]
        degree = self.get_graph().degree[node]

        # k_out = self.out_degrees[node]
        # k_in = degree - k_out
        k_in = self.real_in_degrees[node]
        k_out = self.real_out_degrees[node]

        return H, k_in, k_out

    def get_number_of_nodes(self):
        return self.N

    def get_number_of_communities(self):
        return self.n

    def get_community_sizes(self):
        return self.N_H

    def get_in_degrees(self):
        return self.in_degrees

    def get_out_degrees(self):
        return self.out_degrees

    def get_graph(self):
        return self.graph

    def get_community_per_node(self):
        return self.community_per_node.astype(int)

    def get_p_in(self):
        return self.p_in

    def get_p_out(self):
        return self.p_out

    # create communities as separate graphs in a list
    def create_communities(self):
        communities = []
        counter = 0  # how many node names have been used up to the point
        community_per_node = np.zeros(self.N)
        real_in_degrees = np.empty(0)
        real_in_degrees_per_community = []
        for c in range(self.n):
            deg_sequence = self.in_degrees[c]
            G = nx.empty_graph()
            stublist = to_stublist(deg_sequence)

            # change the numbering of the nodes
            stublist = [node + counter for node in stublist]
            counter += len(deg_sequence)

            if len(stublist) != 0:
                community_per_node[np.min(stublist):np.max(stublist) + 1] = c
            # Choose a random balanced bipartition of the stublist, which
            # gives a random pairing of nodes. In this implementation, we
            # shuffle the list and then split it in half.
            half = len(stublist) // 2
            np.random.shuffle(stublist)
            out_stublist, in_stublist = stublist[:half], stublist[half:]

            G.add_edges_from(zip(out_stublist, in_stublist))

            self_loops = get_self_loops(G)
            for loop in self_loops:
                j = loop[0]
                other_nodes = list(G)
                other_nodes.remove(j)
                new_neighbour = np.random.choice(other_nodes)
                G.add_edge(j, new_neighbour)

            communities.append(G)
            real_in_degrees = np.concatenate((real_in_degrees, np.array(G.degree)[:,1]))
            real_in_degrees_per_community.append(np.array(G.degree)[:,1])
        return communities, community_per_node, real_in_degrees.astype(int), real_in_degrees_per_community

    # join the communities in a single graph
    def join_communities(self):
        # compose communities
        G = nx.empty_graph()
        for c in self.communities:
            G = compose(G, c)

        # connect communities
        stublist = to_stublist(self.out_degrees)

        half = len(stublist) // 2
        np.random.shuffle(stublist)
        out_stublist, in_stublist = stublist[:half], stublist[half:]

        G.add_edges_from(zip(out_stublist, in_stublist))

        return G


def generate_random_HCM(n_com, mu_com, sigma_com, tau_in, m_in, tau_out, m_out):
    """
    n_com : # of communities
    mu_com : avg community size
    sigma_com : community size std
    tau_in : pareto tau parameter for in-degree
    m_in : m pareto parameter  for in-degree
    tau_out : pareto tau parameter for out-degree
    m_out : m pareto parameter  for out-degree
    """

    community_sizes = np.random.normal(mu_com, sigma_com, n_com).astype(int)
    # number of nodes
    N = np.sum(community_sizes)

    intra_community_degrees = []
    for c in community_sizes:
        intra_community_degrees.append(get_pareto_sequence(tau_in, c, m_in))

    inter_community_degrees = get_pareto_sequence(tau_out, N, m_out)

    HCM = HierarchicalConfigurationModel(intra_community_degrees, inter_community_degrees)

    return HCM

if __name__ == '__main__':

    in_degrees = [2*np.ones(100), 2*np.ones(100), 2*np.ones(100)]
    out_degrees = np.ones(300)
    HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)

    max_k_out = HCM.p_out.shape[0] - 1
    max_k_in = HCM.p_in.shape[1] - 1
    n = HCM.get_number_of_communities()
    p_in = HCM.get_p_in()
    p_out = HCM.get_p_out()
    p_out_H = HCM.p_out_H
    N = HCM.N
    N_H = HCM.N_H
    avg_k_in = HCM.avg_k_in
    avg_k_out = HCM.avg_k_out_H
    avg_k_H = avg_k_in + avg_k_out

    p_e_H = np.sum(np.tile(np.arange(0,max_k_out+1),n).reshape(p_out_H.shape)*p_out_H, 1)/avg_k_H
    p_i_H = np.sum(np.tile(np.arange(0,max_k_in+1),n).reshape(p_in.shape)*p_in, 1)/avg_k_H